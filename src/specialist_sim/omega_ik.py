"""Offline client for the Omega-Infra dev-fr3 JSON-lines solver."""

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from specialist_sim import rotation
from specialist_sim.arm import BoundedIk, Fr3, Fr3Layout, IkCfg, IkResult
from specialist_sim.arrays import Float64Array

WORKER = "hal.arm.deploy.robot_control.fr3_replay_worker"


def revision(root: Path) -> str | None:
  if not shutil.which("git"):
    return None
  result = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
  )
  return result.stdout.strip() if result.returncode == 0 else None


def transform(position: Float64Array, orientation: Float64Array) -> Float64Array:
  matrix = np.eye(4)
  matrix[:3, :3] = orientation.reshape(3, 3)
  matrix[:3, 3] = position
  return matrix


def body_transform(arm: Fr3, index: int) -> Float64Array:
  return transform(arm.data.xpos[index], arm.data.xmat[index])


def frames(arm: Fr3, layout: Fr3Layout) -> dict[str, Float64Array]:
  base = np.linalg.inv(body_transform(arm, arm.model.body(layout.base_body).id))
  site = arm.model.site(layout.flange_site).id
  return {
    "flange": base @ transform(arm.data.site_xpos[site], arm.data.site_xmat[site]),
    "tag": base @ body_transform(arm, arm.hand.tag),
  }


def export_chain(arm: Fr3, layout: Fr3Layout, cfg: IkCfg) -> dict[str, Any]:
  # At q=0 each anchor defines the Pinocchio joint origin, including MJCF ref offsets.
  arm.tag_pose(np.zeros_like(arm.home))
  parent = body_transform(arm, arm.model.body(layout.base_body).id)
  base = parent
  joints = []
  if cfg.max_velocity_rad_s.keys() != set(arm.joint_names):
    raise ValueError("max_velocity_rad_s must name each arm joint")
  # MuJoCo numbers joints in tree order. CSV columns may have a different order.
  names = sorted(arm.joint_names, key=lambda n: arm.model.joint(n).id)
  order = [arm.joint_names.index(n) for n in names]
  for name in names:
    jid = arm.model.joint(name).id
    body_rotation = arm.data.xmat[arm.model.jnt_bodyid[jid]].reshape(3, 3)
    joint = transform(arm.data.xanchor[jid], body_rotation)
    joints.append(
      dict(
        name=name,
        placement=(np.linalg.inv(parent) @ joint).tolist(),
        axis=arm.model.jnt_axis[jid].tolist(),
      )
    )
    parent = joint
  end = {
    name: (np.linalg.inv(parent) @ base @ mat).tolist()
    for name, mat in frames(arm, layout).items()
  }
  return dict(
    joints=joints,
    **end,
    home=arm.home[order].tolist(),
    lower=arm.lower[order].tolist(),
    upper=arm.upper[order].tolist(),
    vmax=[cfg.max_velocity_rad_s[name] for name in names],
  )


class WorkerIk:
  def __init__(self, process: subprocess.Popen[str], arm: Fr3, layout: Fr3Layout):
    self.initializer: BoundedIk | None = None
    self.process = process
    self.names = tuple(sorted(arm.joint_names, key=lambda n: arm.model.joint(n).id))
    self.order = [arm.joint_names.index(n) for n in self.names]
    base = arm.model.body(layout.base_body).id
    self.base_from_world = rotation.inverse(
      np.r_[arm.data.xpos[base], arm.data.xquat[base]]
    )

  def request(self, message: dict[str, Any]) -> dict[str, Any]:
    assert self.process.stdin and self.process.stdout
    self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
    self.process.stdin.flush()
    line = self.process.stdout.readline()
    if not line:
      raise RuntimeError(f"IK worker exited ({self.process.poll()}); see worker stderr")
    reply: dict[str, Any] = json.loads(line)
    return reply

  def solve_frame(
    self, target: Float64Array, previous: Float64Array, dt: float | None
  ) -> IkResult:
    if dt is None and self.initializer is not None:
      return self.initializer.solve(target, previous)
    pose = rotation.compose(self.base_from_world, target)
    answer = self.request(
      dict(
        target=transform(pose[:3], rotation.to_matrix(pose[3:])).tolist(),
        previous=previous[self.order].tolist(),
        dt=dt,
      )
    )
    q = None if answer["q"] is None else np.array(answer["q"])
    return IkResult(
      self.names, q, q is not None, answer["status"], tuple(answer["projected"])
    )


def verify_fk(
  solver: WorkerIk, arm: Fr3, layout: Fr3Layout, cfg: IkCfg
) -> dict[str, float]:
  rng = np.random.default_rng(0)
  poses = [
    arm.home,
    arm.lower,
    arm.upper,
    *rng.uniform(arm.lower, arm.upper, (16, len(arm.home))),
  ]
  for i in range(len(arm.home)):
    q = arm.home.copy()
    q[i] = arm.lower[i]
    poses.append(q)
  worst = np.zeros(2)
  for q in poses:
    arm.tag_pose(q)
    reply = solver.request({"fk": q[solver.order].tolist()})
    for name, actual in frames(arm, layout).items():
      reported = np.asarray(reply[name])
      difference = rotation.from_matrix(actual[:3, :3] @ reported[:3, :3].T)
      errors = np.array(
        [
          np.linalg.norm(actual[:3, 3] - reported[:3, 3]),
          np.linalg.norm(rotation.log(difference)),
        ]
      )
      if (
        not np.isfinite(errors).all()
        or (errors > [cfg.fk_position_tolerance_m, cfg.fk_rotation_tolerance_rad]).any()
      ):
        raise ValueError(
          f"Worker FK disagrees with MuJoCo at {name}, q={q}: {errors} (m, rad)"
        )
      worst = np.maximum(worst, errors)
  return dict(max_position_m=float(worst[0]), max_rotation_rad=float(worst[1]))


@contextmanager
def open_worker(
  root: Path,
  python: Path,
  arm: Fr3,
  layout: Fr3Layout,
  cfg: IkCfg,
  limited: bool,
  settings: dict[str, Any],
  *,
  pink: bool = False,
) -> Iterator[tuple[WorkerIk, dict[str, Any]]]:
  if pink and not (root / "pink/solve_ik.py").is_file():
    raise ValueError("--pink-root must point to a Pink checkout")
  entry = (
    [str(Path(__file__).parent / "scripts/pink_worker.py")] if pink else ["-m", WORKER]
  )
  cmd = [str(python.absolute()), "-u", *entry]
  with subprocess.Popen(
    cmd,
    cwd=root,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"},
    text=True,
  ) as proc:
    try:
      chain = export_chain(arm, layout, cfg)
      solver = WorkerIk(proc, arm, layout)
      if pink:
        solver.initializer = BoundedIk(arm, cfg)
      info = solver.request(dict(chain=chain, settings=settings, limited=limited))
      if info["joints"] != list(solver.names):
        raise ValueError("Worker joint order does not match exported chain")
      if info.get("posture_reference_rad") != chain["home"]:
        raise ValueError("IK worker did not confirm the model home posture")
      info.update(revision=revision(root), fk=verify_fk(solver, arm, layout, cfg))
      yield solver, info
    finally:
      if proc.stdin:
        proc.stdin.close()
      if proc.poll() is None:
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()


# Existing Omega callers use the same worker protocol.
open_omega = open_worker
