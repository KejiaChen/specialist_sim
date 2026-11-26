"""Geometry of the tables in the calibrated FR3 replay frame."""

from math import cos, radians, sin, sqrt

import mujoco
import numpy as np
import pytest

from specialist_sim.arm_tables import Fr3Tables
from specialist_sim.gripper import DRAWN

# The example FR3 base faces world -Y. The workcell must follow that heading.
BASE = np.array([0.15, -0.30, -0.57, sqrt(0.5), 0, 0, -sqrt(0.5)])


def test_tables_are_supported_and_aligned_with_the_calibrated_base() -> None:
  spec = mujoco.MjSpec()
  Fr3Tables().stand(spec.worldbody, BASE)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  mount = model.geom("mount_table_top").id
  work = model.geom("work_table_top").id
  floor = model.geom("fr3_floor").id
  ground = data.geom_xpos[floor, 2]
  np.testing.assert_allclose(2 * model.geom_size[mount, :2], [1.20, 0.80])
  np.testing.assert_allclose(2 * model.geom_size[work, :2], [1.20, 0.60])
  heights = data.geom_xpos[[mount, work], 2] + model.geom_size[[mount, work], 2]
  np.testing.assert_allclose(heights - ground, [0.75, 0.94])
  np.testing.assert_allclose(heights[0], BASE[2])
  np.testing.assert_allclose(
    data.geom_xmat[[mount, work]].reshape(2, 3, 3)[:, :, 0],
    [[0, -1, 0], [1, 0, 0]],
    atol=1e-10,
  )
  mount_edge = data.geom_xpos[mount, 1] - model.geom_size[mount, 0]
  work_near = data.geom_xpos[work, 1] + model.geom_size[work, 1]
  np.testing.assert_allclose(mount_edge - work_near, 0.04)
  np.testing.assert_allclose(BASE[1] - 0.07157 - mount_edge, 0.005)
  assert (model.geom_group[[mount, work, floor]] == DRAWN).all()
  assert (model.geom_contype[[mount, work, floor]] != 0).all()
  for name in ("mount_table", "work_table"):
    for suffix in ("00", "01", "10", "11"):
      leg = model.geom(f"{name}_leg_{suffix}").id
      np.testing.assert_allclose(
        data.geom_xpos[leg, 2] - model.geom_size[leg, 2], ground
      )


@pytest.mark.parametrize("work_x", [None, 1.20])
def test_work_table_can_rotate_and_translate_in_mount_coordinates(
  work_x: float | None,
) -> None:
  spec = mujoco.MjSpec()
  Fr3Tables(work_x=work_x, work_y=0.08, work_yaw_deg=75).stand(spec.worldbody, BASE)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  mount = model.body("mount_table").id
  work = model.body("work_table").id
  yaw = radians(75)
  expected_x = 0.64 + 0.60 * cos(yaw) + 0.30 * sin(yaw) if work_x is None else work_x
  np.testing.assert_allclose(
    (data.xpos[work] - data.xpos[mount])[:2], [0.08, -expected_x]
  )
  np.testing.assert_allclose(
    data.xmat[work].reshape(3, 3)[:, 0], [sin(yaw), -cos(yaw), 0]
  )
