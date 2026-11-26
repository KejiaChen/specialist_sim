"""The tables and floor that surround the FR3."""

from dataclasses import dataclass
from math import atan2, cos, isfinite, radians, sin

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import DRAWN

_MOUNT_SIZE = (1.20, 0.80)
_WORK_SIZE = (1.20, 0.60)
_MOUNT_HEIGHT = 0.75
_WORK_HEIGHT = 0.94
_TOP_THICKNESS = 0.03
_WORK_GAP = 0.04
# Leading edge of the Menagerie FR3 link0 collision mesh in the base frame.
_BASE_FRONT = 0.07157
_BASE_CLEARANCE = 0.005


@dataclass(frozen=True)
class Fr3Tables:
  """The work table pose in the mounting-table frame, in meters and degrees."""

  work_x: float | None = None
  work_y: float = 0.0
  work_yaw_deg: float = 90.0

  def __post_init__(self) -> None:
    if not all(isfinite(value) for value in (self.work_y, self.work_yaw_deg)) or (
      self.work_x is not None and not isfinite(self.work_x)
    ):
      raise ValueError("table positions and yaw must be finite")

  def stand(self, world: mujoco.MjsBody, base: Float64Array) -> None:
    """Build the scene around the calibrated FR3 base in replay coordinates."""
    facing = rotation.rotate(base[3:], np.array([1.0, 0.0, 0.0]))
    yaw = atan2(facing[1], facing[0])
    forward = np.array([cos(yaw), sin(yaw), 0.0])
    left = np.array([-sin(yaw), cos(yaw), 0.0])
    mount_edge = _MOUNT_SIZE[0] / 2
    mount_top = base[:3] - (mount_edge - _BASE_FRONT - _BASE_CLEARANCE) * forward
    work_yaw = radians(self.work_yaw_deg)
    work_x = self.work_x
    if work_x is None:
      length, width = _WORK_SIZE
      half_extent = (length * abs(cos(work_yaw)) + width * abs(sin(work_yaw))) / 2
      work_x = mount_edge + _WORK_GAP + half_extent
    work_top = mount_top + work_x * forward + self.work_y * left
    work_top[2] += _WORK_HEIGHT - _MOUNT_HEIGHT
    world.add_geom(
      name="fr3_floor",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      pos=[base[0], base[1], base[2] - _MOUNT_HEIGHT],
      size=[0, 0, 0.05],
      group=DRAWN,
      rgba=[0.29, 0.32, 0.35, 1],
    )
    _table(
      world,
      "mount_table",
      mount_top,
      _MOUNT_SIZE,
      _MOUNT_HEIGHT,
      yaw,
      (0.67, 0.52, 0.36, 1),
    )
    _table(
      world,
      "work_table",
      work_top,
      _WORK_SIZE,
      _WORK_HEIGHT,
      yaw + work_yaw,
      (0.54, 0.67, 0.68, 1),
    )


def _table(
  world: mujoco.MjsBody,
  name: str,
  top: Float64Array,
  size: tuple[float, float],
  height: float,
  yaw: float,
  color: tuple[float, float, float, float],
) -> None:
  table = world.add_body(name=name, pos=top, quat=[cos(yaw / 2), 0, 0, sin(yaw / 2)])
  half_x, half_y = (side / 2 for side in size)
  table.add_geom(
    name=f"{name}_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0, 0, -_TOP_THICKNESS / 2],
    size=[half_x, half_y, _TOP_THICKNESS / 2],
    group=DRAWN,
    rgba=color,
  )
  half_leg = (height - _TOP_THICKNESS) / 2
  for i, x in enumerate((-half_x + 0.06, half_x - 0.06)):
    for j, y in enumerate((-half_y + 0.06, half_y - 0.06)):
      table.add_geom(
        name=f"{name}_leg_{i}{j}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[x, y, -_TOP_THICKNESS - half_leg],
        size=[0.025, 0.025, half_leg],
        group=DRAWN,
        rgba=[0.20, 0.23, 0.26, 1],
      )
