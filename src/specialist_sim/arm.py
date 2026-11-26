"""Kinematic replay of a Specialist gripper mounted on a Franka FR3."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import mujoco
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from specialist_sim import gripper, rotation
from specialist_sim.arm_tables import Fr3Tables
from specialist_sim.arrays import Float64Array
from specialist_sim.camera.lens import Lens
from specialist_sim.env import Cfg, Lighting
from specialist_sim.take import HandState


@dataclass(frozen=True)
class Fr3Layout:
  model: Path
  world_from_base: tuple[float, ...]
  label: str
  attachment_from_tag: tuple[float, ...] | None = None
  gripper_mjcf: Path | None = None
  attachment_from_gripper_base: tuple[float, ...] | None = None
  gripper_base_from_tag: tuple[float, ...] | None = None
  gripper_opening_at_zero: float = 0.0
  base_body: str = "base"
  flange_body: str = "fr3_link7"
  flange_site: str = "attachment_site"
  model_name: str = "fr3"
  home_key: str = "home"
  joints: tuple[str, ...] = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
  )
  visual_group: int = 2
  tables: Fr3Tables | None = None

  def mount(self) -> tuple[str, Float64Array]:
    if self.gripper_mjcf is None:
      if self.attachment_from_tag is None:
        raise ValueError("the original gripper requires attachment_from_tag")
      if (
        self.attachment_from_gripper_base is not None
        or self.gripper_base_from_tag is not None
      ):
        raise ValueError("gripper base transforms require gripper_mjcf")
      return gripper.TAG_BODY, _pose(self.attachment_from_tag, "attachment_from_tag")
    if (
      self.attachment_from_tag is not None or self.attachment_from_gripper_base is None
    ):
      raise ValueError(
        "gripper_mjcf requires attachment_from_gripper_base instead of attachment_from_tag"
      )
    return gripper.BASE_BODY, _pose(
      self.attachment_from_gripper_base, "attachment_from_gripper_base"
    )


@dataclass(frozen=True)
class IkCfg:
  max_velocity_rad_s: dict[str, float] = field(
    default_factory=lambda: {
      "fr3_joint1": 2.62,
      "fr3_joint2": 2.62,
      "fr3_joint3": 2.62,
      "fr3_joint4": 2.62,
      "fr3_joint5": 5.26,
      "fr3_joint6": 4.18,
      "fr3_joint7": 5.26,
    }
  )
  velocity_tolerance_rad_s: float = 1e-8
  fk_position_tolerance_m: float = 1e-6
  fk_rotation_tolerance_rad: float = 1e-6
  position_tolerance_m: float = 0.005
  rotation_tolerance_deg: float = 3.0
  near_limit_deg: float = 2.0
  max_evaluations: int = 35
  position_weight: float = 1.0
  rotation_weight: float = 1.0
  home_weight: float = 0.02
  continuity_weight: float = 0.1
  min_penetration_m: float = 0.0

  def __post_init__(self) -> None:
    positive = (
      *self.max_velocity_rad_s.values(),
      self.velocity_tolerance_rad_s,
      self.fk_position_tolerance_m,
      self.fk_rotation_tolerance_rad,
      self.position_tolerance_m,
      self.rotation_tolerance_deg,
      self.near_limit_deg,
      self.max_evaluations,
      self.position_weight,
      self.rotation_weight,
    )
    nonnegative = (self.home_weight, self.continuity_weight, self.min_penetration_m)
    if type(self.max_evaluations) is not int:
      raise ValueError("max_evaluations must be an integer")
    if not all(np.isfinite(v) and v > 0 for v in positive) or not all(
      np.isfinite(v) and v >= 0 for v in nonnegative
    ):
      raise ValueError("IK tolerances, budget and weights must be finite and valid")


def _pose(values: tuple[float, ...], name: str) -> Float64Array:
  pose = np.asarray(values, dtype=np.float64)
  if pose.shape != (7,) or not np.isfinite(pose).all():
    raise ValueError(f"{name}: expected [x, y, z, qw, qx, qy, qz]")
  norm = float(np.linalg.norm(pose[3:]))
  if not np.isfinite(norm) or norm == 0.0:
    raise ValueError(f"{name}: expected a nonzero quaternion")
  return np.concatenate((pose[:3], pose[3:] / norm))


class Fr3:
  def __init__(self, layout: Fr3Layout, hand: str) -> None:
    spec = mujoco.MjSpec.from_file(str(layout.model))
    if spec.modelname != layout.model_name:
      raise ValueError(
        f"{layout.model}: expected {layout.model_name}, got {spec.modelname}"
      )
    arm_joints = {joint.name for joint in spec.joints}
    if len(layout.joints) != len(arm_joints) or set(layout.joints) != arm_joints:
      raise ValueError(f"{layout.model}: expected FR3 joints {layout.joints}")
    base = _pose(layout.world_from_base, "world_from_base")
    mount_body, mount = layout.mount()
    base_from_tag = (
      None
      if layout.gripper_base_from_tag is None
      else _pose(layout.gripper_base_from_tag, "gripper_base_from_tag")
    )
    if not np.isfinite(layout.gripper_opening_at_zero):
      raise ValueError("gripper_opening_at_zero must be finite")
    spec.body(layout.base_body).pos = base[:3]
    spec.body(layout.base_body).quat = base[3:]
    if layout.tables is not None:
      layout.tables.stand(spec.worldbody, base)
    site = spec.site(layout.flange_site)
    if site.parent.name != layout.flange_body:
      raise ValueError(f"{layout.flange_site} must be attached to {layout.flange_body}")
    link_from_mount = rotation.compose(np.concatenate((site.pos, site.quat)), mount)
    carrier = spec.body(layout.flange_body).add_body(
      name=f"{hand}_mount", pos=link_from_mount[:3], quat=link_from_mount[3:]
    )
    gripper.attach(
      spec,
      carrier,
      hand,
      Lens(),
      Cfg().finger,
      model_path=layout.gripper_mjcf,
      mount_body=mount_body,
      base_from_tag=base_from_tag,
    )
    Lighting().stand(spec)

    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    self.hand = gripper.bind(self.model, hand)
    self.joint_names = layout.joints

    joint_ids = np.array(
      [self.model.joint(name).id for name in layout.joints], dtype=int
    )
    if (
      not len(joint_ids)
      or not self.model.jnt_limited[joint_ids].all()
      or not (self.model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE).all()
    ):
      raise ValueError("FR3 needs unique, limited hinge joints")
    self._qpos = self.model.jnt_qposadr[joint_ids]
    self.lower, self.upper = self.model.jnt_range[joint_ids].T.copy()
    self.home = self.model.key(layout.home_key).qpos[self._qpos].copy()
    if not self.in_bounds(self.home):
      raise ValueError("FR3 home is outside its joint limits")

    fingers = [
      self.model.joint(f"{hand}_gripper_{side}_finger").id for side in gripper.HANDS
    ]
    left, right = self.model.jnt_range[fingers]
    zero = self._opening_at_zero = layout.gripper_opening_at_zero
    lo, hi = gripper.SHUT, gripper.OPEN
    if self.model.jnt_limited[fingers[0]]:
      lo, hi = max(lo, left[0] + zero), min(hi, left[1] + zero)
    if self.model.jnt_limited[fingers[1]]:
      lo, hi = max(lo, -right[1] + zero), min(hi, -right[0] + zero)
    if lo > hi:
      raise ValueError(
        "gripper finger limits do not overlap the recording's opening range"
      )
    self._opening_range = (lo, hi)
    self._finger_qpos = list(self.hand.fingers)
    # Descendants share the arm root, so classify collision geometry by body.
    self._on_gripper = np.array(
      [
        self.model.body(i).name.startswith(f"{hand}_gripper_")
        for i in self.model.geom_bodyid
      ],
      dtype=bool,
    )
    if layout.tables is not None:
      self.place(self.home, gripper.OPEN)

  def in_bounds(self, q: Float64Array) -> bool:
    return bool(
      q.shape == self.home.shape
      and np.isfinite(q).all()
      and (q >= self.lower).all()
      and (q <= self.upper).all()
    )

  def placed_tag_pose(self) -> Float64Array:
    """Read the tag after setting the model coordinates and computing FK."""
    return np.concatenate(
      (self.data.xpos[self.hand.tag], self.data.xquat[self.hand.tag])
    )

  def tag_pose(self, q: Float64Array) -> Float64Array:
    self.data.qpos[self._qpos] = q
    mujoco.mj_kinematics(self.model, self.data)
    return self.placed_tag_pose()

  def place(self, q: Float64Array, opening: float) -> float:
    applied = float(np.clip(opening, *self._opening_range))
    self.data.qpos[self._qpos] = q
    angle = applied - self._opening_at_zero
    self.data.qpos[self._finger_qpos] = (angle, -angle)
    mujoco.mj_fwdPosition(self.model, self.data)
    return applied

  def collision_pairs(self, min_penetration_m: float) -> tuple[tuple[str, str], ...]:
    pairs = set()
    for contact in self.data.contact:
      a, b = (int(g) for g in contact.geom)
      if contact.dist >= -min_penetration_m or (
        self._on_gripper[a] and self._on_gripper[b]
      ):
        continue
      names = [
        self.model.geom(g).name or self.model.body(self.model.geom_bodyid[g]).name
        for g in (a, b)
      ]
      pairs.add((min(names), max(names)))
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class IkResult:
  joints: tuple[str, ...]
  q: Float64Array | None
  converged: bool
  status: str
  projected: tuple[str, ...] = ()


@runtime_checkable
class TimedSolver(Protocol):
  def solve_frame(
    self, target: Float64Array, previous: Float64Array, dt: float | None
  ) -> IkResult: ...


class Solver(Protocol):
  def solve(self, target: Float64Array, previous: Float64Array) -> IkResult:
    """Find joint positions for a world-frame base-tag target."""


def pose_error(
  actual: Float64Array, target: Float64Array
) -> tuple[Float64Array, Float64Array]:
  return (
    actual[:3] - target[:3],
    rotation.log(rotation.multiply(actual[3:], rotation.conjugate(target[3:]))),
  )


class BoundedIk:
  def __init__(self, arm: Fr3, cfg: IkCfg) -> None:
    self.arm, self.cfg = arm, cfg
    self._scales = (
      np.sqrt(cfg.position_weight) / cfg.position_tolerance_m,
      np.sqrt(cfg.rotation_weight) / np.deg2rad(cfg.rotation_tolerance_deg),
      np.sqrt(cfg.home_weight),
      np.sqrt(cfg.continuity_weight),
    )

  def solve(self, target: Float64Array, previous: Float64Array) -> IkResult:
    pos, ang, home, smooth = self._scales

    def residual(q: Float64Array) -> Float64Array:
      p, r = pose_error(self.arm.tag_pose(q), target)
      return np.concatenate(
        (pos * p, ang * r, home * (q - self.arm.home), smooth * (q - previous))
      )

    try:
      result = least_squares(
        residual,
        previous,
        bounds=(self.arm.lower, self.arm.upper),
        max_nfev=self.cfg.max_evaluations,
      )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
      return IkResult(self.arm.joint_names, None, False, str(exc))
    return IkResult(
      self.arm.joint_names, result.x, bool(result.success), str(result.message)
    )


@dataclass(frozen=True)
class Frame:
  time: float
  target: Float64Array
  actual: Float64Array
  q: Float64Array
  requested_opening: float
  opening: float
  position_error_m: float
  rotation_error_deg: float
  pose_ok: bool
  converged: bool
  status: str
  held_previous: bool
  near_limits: tuple[str, ...]
  collisions: tuple[tuple[str, str], ...]

  dt: float | None
  velocity: Float64Array | None
  at_speed_limit: tuple[str, ...]
  over_speed_limit: tuple[str, ...]
  projected: tuple[str, ...]

  @property
  def passed(self) -> bool:
    return (
      self.converged and self.pose_ok and not self.held_previous and not self.collisions
    )


def replay(
  arm: Fr3, solver: Solver | TimedSolver, track: HandState, cfg: IkCfg
) -> Iterator[Frame]:
  if not np.isfinite(track.times).all() or np.any(np.diff(track.times) <= 0):
    raise ValueError("frame timestamps must be finite and strictly increasing")
  if set(arm.joint_names) != cfg.max_velocity_rad_s.keys():
    raise ValueError("max_velocity_rad_s must name each arm joint")
  vmax = np.array([cfg.max_velocity_rad_s[n] for n in arm.joint_names])
  last_output = previous = arm.home
  last_time: float | None = None
  for instant, target, requested in zip(
    track.times, track.poses, track.opening, strict=True
  ):
    dt = None if last_time is None else float(instant) - last_time
    if isinstance(solver, TimedSolver):
      previous = last_output
      result = solver.solve_frame(target, previous, dt)
    else:
      result = solver.solve(target, previous)
    if set(result.joints) != set(arm.joint_names) or len(result.joints) != len(
      arm.joint_names
    ):
      raise ValueError("solver result must name each FR3 joint once")
    q = result.q
    if q is not None and q.shape == arm.home.shape:
      q = q[[result.joints.index(name) for name in arm.joint_names]]
    if q is None or not arm.in_bounds(q):
      q, held = previous, True
    else:
      held = False

    opening = arm.place(q, float(requested))
    actual = arm.placed_tag_pose()
    p, r = pose_error(actual, target)
    pe, re = float(np.linalg.norm(p)), float(np.rad2deg(np.linalg.norm(r)))
    pose_ok = pe <= cfg.position_tolerance_m and re <= cfg.rotation_tolerance_deg
    if result.converged and pose_ok and not held:
      previous = q
    margin = np.rad2deg(np.minimum(q - arm.lower, arm.upper - q))

    velocity = None if dt is None else (q - last_output) / dt
    speed = np.zeros_like(q) if velocity is None else np.abs(velocity)
    last_time, last_output = float(instant), q
    yield Frame(
      float(instant),
      target,
      actual,
      q,
      float(requested),
      opening,
      pe,
      re,
      pose_ok,
      result.converged,
      result.status,
      held,
      tuple(
        name
        for name, value in zip(arm.joint_names, margin, strict=True)
        if value <= cfg.near_limit_deg
      ),
      arm.collision_pairs(cfg.min_penetration_m),
      dt,
      velocity,
      tuple(
        n
        for n, hit in zip(
          arm.joint_names,
          np.abs(speed - vmax) <= cfg.velocity_tolerance_rad_s,
          strict=True,
        )
        if hit
      ),
      tuple(
        n
        for n, over in zip(
          arm.joint_names, speed > vmax + cfg.velocity_tolerance_rad_s, strict=True
        )
        if over
      ),
      result.projected,
    )
