"""Cross-environment integration checks with the user's FR3 + Omega layout."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("specialist_dataset")
pytest.importorskip("scipy")

from specialist_sim.arm import Fr3, IkResult, replay  # noqa: E402
from specialist_sim.arrays import Float64Array  # noqa: E402
from specialist_sim.omega_ik import open_omega  # noqa: E402
from specialist_sim.scripts.replay_arm import read_config  # noqa: E402
from specialist_sim.take import HandState  # noqa: E402


def test_fr3_solver_and_failure_hold() -> None:
  keys = ("FR3_REPLAY_CONFIG", "OMEGA_ROOT", "OMEGA_PYTHON")
  if not all(os.environ.get(key) for key in keys):
    pytest.skip("set FR3_REPLAY_CONFIG, OMEGA_ROOT and OMEGA_PYTHON")
  config, root, python = (Path(os.environ[key]) for key in keys)
  layout, cfg, _ = read_config(config, None)
  layout = replace(layout, joints=tuple(reversed(layout.joints)))
  arm = Fr3(layout, "right")
  home = arm.tag_pose(arm.home)
  far = home.copy()
  far[0] += 3
  track = HandState(
    np.array([0.0, 0.02, 0.021, 0.04]), np.stack([home, far, far, home]), np.zeros(4)
  )
  with open_omega(root, python, arm, layout, cfg, True, {}) as (solver, _):

    class FailThird:
      calls = 0

      def solve_frame(
        self, target: Float64Array, previous: Float64Array, dt: float | None
      ) -> IkResult:
        self.calls += 1
        if self.calls == 3:
          return IkResult(arm.joint_names, None, False, "forced failure")
        return solver.solve_frame(target, previous, dt)

    frames = list(replay(arm, FailThird(), track, cfg))
  assert frames[0].converged
  assert frames[0].pose_ok
  assert not frames[1].pose_ok and not frames[1].held_previous
  assert frames[2].held_previous and frames[3].converged
  np.testing.assert_array_equal(frames[2].q, frames[1].q)
  speeds = abs(np.diff([f.q for f in frames], axis=0) / np.diff(track.times)[:, None])
  assert (
    speeds
    <= np.array([cfg.max_velocity_rad_s[n] for n in layout.joints])
    + cfg.velocity_tolerance_rad_s
  ).all()
  with pytest.raises(ValueError, match="timestamps"):
    list(
      replay(arm, FailThird(), replace(track, times=np.zeros_like(track.times)), cfg)
    )
