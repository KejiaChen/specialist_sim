"""Replay one reconstructed gripper trajectory through a Franka FR3."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from specialist_dataset import schema
from specialist_dataset.layer import encode
from specialist_dataset.read import Recording

from specialist_sim import gripper, omega_ik, video
from specialist_sim.arm import BoundedIk, Fr3, Fr3Layout, Frame, IkCfg, replay
from specialist_sim.arrays import Float64Array, UInt8Array
from specialist_sim.take import hand_state, sole_state_layer

PREVIEW_SUFFIXES = (".jpg", ".jpeg")
PREVIEW_OPEN_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class VideoCfg:
  width: int = 960
  height: int = 720
  fps: float = 30.0
  distance: float = 1.65
  azimuth: float = 142.0
  elevation: float = -30.0
  target_radius: float = 0.014
  actual_radius: float = 0.008

  def __post_init__(self) -> None:
    if any(type(n) is not int or n <= 0 or n % 2 for n in (self.width, self.height)):
      raise ValueError("video width and height must be positive, even pixel counts")
    if not np.isfinite(self.fps) or self.fps <= 0:
      raise ValueError("video fps must be positive")


def read_config(path: Path, model: Path | None) -> tuple[Fr3Layout, IkCfg, VideoCfg]:
  with path.open("rb") as file:
    config = tomllib.load(file)
  layout = dict(config["layout"])
  declared = layout.pop("model", None)
  if model is None and declared is None:
    raise ValueError("give layout.model in the config or pass --model")
  asset = model if model is not None else path.parent / declared
  layout["world_from_base"] = tuple(layout["world_from_base"])
  for field in (
    "attachment_from_tag",
    "attachment_from_gripper_base",
    "gripper_base_from_tag",
  ):
    if field in layout:
      layout[field] = tuple(layout[field])
  if "gripper_mjcf" in layout:
    layout["gripper_mjcf"] = (path.parent / layout["gripper_mjcf"]).resolve(strict=True)
  if "joints" in layout:
    layout["joints"] = tuple(layout["joints"])
  return (
    Fr3Layout(model=asset.resolve(strict=True), **layout),
    IkCfg(**config.get("ik", {})),
    VideoCfg(**config.get("video", {})),
  )


def write_frames(path: Path, frames: list[Frame], layout: Fr3Layout) -> None:
  pose_labels = ("x", "y", "z", "qw", "qx", "qy", "qz")
  with path.open("w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(
      [
        "time_s",
        *layout.joints,
        "requested_opening_rad",
        "opening_rad",
        "position_error_m",
        "rotation_error_deg",
        "pose_ok",
        "solver_converged",
        "solver_status",
        "held_previous",
        "near_limits",
        "collision_pairs",
        "passed",
        *(f"target_{x}" for x in pose_labels),
        *(f"actual_{x}" for x in pose_labels),
        "dt_s",
        *(f"{name}_velocity_rad_s" for name in layout.joints),
        "at_speed_limit",
        "over_speed_limit",
        "projected_joints",
      ]
    )
    writer.writerows(
      (
        f.time,
        *f.q,
        f.requested_opening,
        f.opening,
        f.position_error_m,
        f.rotation_error_deg,
        int(f.pose_ok),
        int(f.converged),
        f.status,
        int(f.held_previous),
        json.dumps(f.near_limits),
        json.dumps(f.collisions),
        int(f.passed),
        *f.target,
        *f.actual,
        f.dt,
        *(f.velocity if f.velocity is not None else [None] * len(layout.joints)),
        json.dumps(f.at_speed_limit),
        json.dumps(f.over_speed_limit),
        json.dumps(f.projected),
      )
      for f in frames
    )


def failed_spans(frames: list[Frame]) -> list[dict[str, float | int]]:
  bad = np.array([not frame.passed for frame in frames])
  starts, stops = np.flatnonzero(np.diff(np.r_[False, bad, False])).reshape(-1, 2).T
  return [
    {
      "start_s": frames[start].time,
      "end_s": frames[stop - 1].time,
      "frames": int(stop - start),
    }
    for start, stop in zip(starts, stops, strict=True)
  ]


def statistics(values: list[float]) -> dict[str, float]:
  return dict(
    zip(("median", "p95", "max"), np.percentile(values, (50, 95, 100)), strict=True)
  )


def summary(frames: list[Frame], layout: Fr3Layout, ik: IkCfg) -> dict[str, Any]:
  passed = sum(frame.passed for frame in frames)
  return {
    "layout": asdict(layout),
    "ik": asdict(ik),
    "frames": len(frames),
    "time_range_s": [frames[0].time, frames[-1].time],
    "passed": passed,
    "pass_rate": passed / len(frames),
    "velocity_abs_rad_s": {
      name: statistics(
        [abs(float(f.velocity[i])) for f in frames if f.velocity is not None]
      )
      for i, name in enumerate(layout.joints)
    }
    if len(frames) > 1
    else {},
    "over_speed_frames": sum(bool(f.over_speed_limit) for f in frames),
    "at_speed_limit_frames": sum(bool(f.at_speed_limit) for f in frames),
    "filter_projection_frames": sum(bool(f.projected) for f in frames),
    "pose_ok": sum(frame.pose_ok for frame in frames),
    "solver_status": Counter(frame.status for frame in frames),
    "solver_not_converged": sum(not frame.converged for frame in frames),
    "held_previous": sum(frame.held_previous for frame in frames),
    "opening_clipped": sum(
      frame.requested_opening != frame.opening for frame in frames
    ),
    "position_error_m": statistics([frame.position_error_m for frame in frames]),
    "rotation_error_deg": statistics([frame.rotation_error_deg for frame in frames]),
    "near_limit_frames": Counter(n for frame in frames for n in frame.near_limits),
    "collision_frames": sum(bool(frame.collisions) for frame in frames),
    "collision_pairs": [
      {"geoms": pair, "frames": count}
      for pair, count in Counter(
        p for frame in frames for p in frame.collisions
      ).items()
    ],
    "failed_spans": failed_spans(frames),
  }


def _marker(
  scene: mujoco.MjvScene, position: Float64Array, radius: float, rgba: list[float]
) -> None:
  mujoco.mjv_initGeom(
    scene.geoms[scene.ngeom],
    mujoco.mjtGeom.mjGEOM_SPHERE,
    [radius] * 3,
    position,
    np.eye(3).ravel(),
    rgba,
  )
  scene.ngeom += 1


@contextmanager
def frame_renderer(
  arm: Fr3, layout: Fr3Layout, frames: Sequence[Frame], cfg: VideoCfg
) -> Iterator[Callable[[Frame], UInt8Array]]:
  camera = mujoco.MjvCamera()
  camera.lookat[:] = (
    np.mean([f.target[:3] for f in frames], axis=0)
    + np.array(layout.world_from_base[:3])
  ) / 2
  camera.distance, camera.azimuth, camera.elevation = (
    cfg.distance,
    cfg.azimuth,
    cfg.elevation,
  )
  option = mujoco.MjvOption()
  option.geomgroup[:] = False
  option.geomgroup[[gripper.DRAWN, layout.visual_group]] = True
  option.sitegroup[:] = False

  arm.model.vis.global_.offwidth = cfg.width
  arm.model.vis.global_.offheight = cfg.height
  with mujoco.Renderer(arm.model, cfg.height, cfg.width) as viewer:

    def draw(frame: Frame) -> UInt8Array:
      arm.place(frame.q, frame.opening)
      viewer.update_scene(arm.data, camera, scene_option=option)
      _marker(viewer.scene, frame.target[:3], cfg.target_radius, [0.1, 0.65, 1.0, 0.45])
      color = [0.2, 0.9, 0.2, 1.0] if frame.passed else [0.95, 0.2, 0.12, 1.0]
      _marker(viewer.scene, frame.actual[:3], cfg.actual_radius, color)
      return np.asarray(viewer.render(), dtype=np.uint8)

    yield draw


def render(
  path: Path, arm: Fr3, layout: Fr3Layout, frames: list[Frame], cfg: VideoCfg
) -> None:
  times = np.array([frame.time for frame in frames])
  instants = np.arange(times[0], times[-1] + np.finfo(float).eps, 1 / cfg.fps)
  picked = np.searchsorted(times, instants, side="right") - 1
  with (
    frame_renderer(arm, layout, frames, cfg) as draw,
    video.writer(path, cfg.width, cfg.height, cfg.fps) as write,
  ):
    for i in picked:
      write(draw(frames[i]))


def preview_first(
  path: Path, arm: Fr3, layout: Fr3Layout, frame: Frame, cfg: VideoCfg
) -> None:
  with frame_renderer(arm, layout, (frame,), cfg) as draw:
    jpeg = encode(draw(frame))
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=path.parent) as directory:
    staged = Path(directory) / path.name
    staged.write_bytes(jpeg)
    staged.replace(path)
  print(
    f"First point at {frame.time:.3f} s: {'passed' if frame.passed else 'failed'}, "
    f"{frame.position_error_m * 1000:.3f} mm, {frame.rotation_error_deg:.3f} deg, "
    f"{len(frame.collisions)} collision pairs. Wrote {path}"
  )


def open_preview(path: Path) -> None:
  """Open the saved JPEG with the Linux desktop's image viewer, if available."""
  if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    return
  opener = shutil.which("xdg-open")
  if opener is None:
    print("Could not open preview: xdg-open is unavailable", file=sys.stderr)
    return
  try:
    viewer = subprocess.Popen(
      [opener, str(path.resolve())],
      stdin=subprocess.DEVNULL,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
    status = viewer.wait(timeout=PREVIEW_OPEN_TIMEOUT_S)
  except subprocess.TimeoutExpired:
    # Some handlers stay open until the viewer window closes.
    return
  except OSError as exc:
    print(f"Could not open preview: {exc}", file=sys.stderr)
    return
  if status:
    print(
      f"Could not open preview: xdg-open exited with status {status}", file=sys.stderr
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--session", required=True)
  parser.add_argument("--layer", help="which state layer to read")
  parser.add_argument("--hand", choices=gripper.HANDS, default="right")
  parser.add_argument(
    "--config", type=Path, required=True, help="FR3 installation TOML"
  )
  parser.add_argument("--model", type=Path, help="override the FR3 MJCF path")
  parser.add_argument(
    "--solver", choices=("scipy", "omega", "omega-limited"), default="scipy"
  )
  parser.add_argument("--omega-root", type=Path, help="Omega-Infra dev-fr3 checkout")
  parser.add_argument(
    "--omega-python", type=Path, help="Python executable with the Omega IK environment"
  )
  output = parser.add_mutually_exclusive_group(required=True)
  output.add_argument("--output-dir", type=Path, help="write full trajectory reports")
  output.add_argument(
    "--preview-first",
    type=Path,
    metavar="IMAGE.jpg",
    help="solve only the first trajectory point and save or replace its JPEG preview",
  )
  parser.add_argument("--video", action="store_true", help="render an overview MP4")
  parser.add_argument(
    "--no-preview-window",
    action="store_true",
    help="save the first-point preview without opening the desktop viewer",
  )
  args = parser.parse_args()
  if args.preview_first is not None:
    if args.video:
      parser.error("--video requires --output-dir")
    if args.preview_first.suffix.lower() not in PREVIEW_SUFFIXES:
      parser.error(
        f"--preview-first requires an image path ending in {', '.join(PREVIEW_SUFFIXES)}"
      )
  try:
    layout, ik, video_cfg = read_config(args.config, args.model)
    layer = args.layer or sole_state_layer(args.dataset)
    state_path = schema.layer_rrd(args.dataset, layer, args.session)
    if not state_path.is_file():
      raise FileNotFoundError(state_path)
    with Recording(state_path) as recording:
      track = hand_state(recording, args.hand)
    arm = Fr3(layout, args.hand)
  except (KeyError, OSError, TypeError, ValueError) as exc:
    parser.error(str(exc))

  names = ["frames.csv", "summary.json"]
  if args.video:
    names.append("replay.mp4")
  if args.output_dir is not None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if present := [name for name in names if (args.output_dir / name).exists()]:
      parser.error(f"{args.output_dir}: outputs already exist: {', '.join(present)}")

  solver_info: dict[str, Any] = {}
  if args.preview_first is not None:
    track = type(track)(track.times[:1], track.poses[:1], track.opening[:1])
  print(
    f"{args.session} {args.hand}: solving {len(track.times)} state rows with {args.solver}",
    flush=True,
  )
  if args.solver == "scipy":
    import scipy  # type: ignore[import-untyped]

    solver_info["scipy"] = scipy.__version__
    frames = list(replay(arm, BoundedIk(arm, ik), track, ik))
  else:
    if args.omega_root is None or args.omega_python is None:
      parser.error("Omega requires --omega-root and --omega-python")
    with args.config.open("rb") as file:
      settings = tomllib.load(file).get("omega", {})
    try:
      with omega_ik.open_omega(
        args.omega_root,
        args.omega_python,
        arm,
        layout,
        ik,
        args.solver == "omega-limited",
        settings,
      ) as (solver, solver_info):
        frames = list(replay(arm, solver, track, ik))
    except (ValueError, RuntimeError, OSError) as exc:
      parser.error(f"{args.omega_python} in {args.omega_root}: {exc}")
  if args.preview_first is not None:
    preview_first(args.preview_first, arm, layout, frames[0], video_cfg)
    if not args.no_preview_window:
      open_preview(args.preview_first)
    return

  report = summary(frames, layout, ik)
  report.update(
    solver=args.solver,
    solver_info=solver_info,
    sim_revision=omega_ik.revision(Path(__file__).resolve().parent),
    session=args.session,
    hand=args.hand,
    state_path=str(state_path.resolve()),
    model_sha256=sha256(layout.model.read_bytes()).hexdigest(),
    mujoco_version=mujoco.__version__,
  )
  if layout.gripper_mjcf is not None:
    report["gripper_mjcf_sha256"] = sha256(layout.gripper_mjcf.read_bytes()).hexdigest()

  with tempfile.TemporaryDirectory(dir=args.output_dir) as directory:
    staging = Path(directory)
    write_frames(staging / "frames.csv", frames, layout)
    (staging / "summary.json").write_text(
      json.dumps(report, indent=2, default=str) + "\n"
    )
    if args.video:
      print("Rendering the FR3 replay", flush=True)
      render(staging / "replay.mp4", arm, layout, frames, video_cfg)
    for name in names:
      (staging / name).replace(args.output_dir / name)

  print(
    f"{report['passed']}/{len(frames)} rows passed, "
    f"{report['collision_frames']} collision rows. Wrote {args.output_dir}"
  )


if __name__ == "__main__":
  main()
