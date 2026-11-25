"""Run as a standalone script from a Pink checkout using its Python environment."""

import json
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore[import-not-found]

# Load the selected checkout rather than another installed copy.
sys.path.insert(0, str(Path.cwd()))
import pink  # type: ignore[import-not-found] # noqa: E402
import pink.exceptions  # type: ignore[import-not-found] # noqa: E402
import pink.tasks  # type: ignore[import-not-found] # noqa: E402


@dataclass(frozen=True)
class Settings:
  position_cost: float = 1.0
  orientation_cost: float = 1.0
  lm_damping: float = 1.0
  posture_cost: float = 1e-3
  damping_cost: float = 1e-3

  def __post_init__(self) -> None:
    if not all(np.isfinite(v) and v >= 0 for v in asdict(self).values()):
      raise ValueError("Pink task costs and damping must be finite and nonnegative")


def placement(matrix: Any) -> Any:
  mat = np.asarray(matrix)
  return pin.SE3(mat[:3, :3], mat[:3, 3])


def reply(**fields: Any) -> None:
  print(json.dumps(fields, allow_nan=False), flush=True)


def main() -> None:
  init = json.loads(input())
  chain = init["chain"]
  settings = Settings(**init["settings"])
  model = pin.Model()
  parent = 0
  for joint in chain["joints"]:
    parent = model.addJoint(
      parent,
      pin.JointModelRevoluteUnaligned(np.array(joint["axis"])),
      placement(joint["placement"]),
      joint["name"],
    )
  for name in ("tag", "flange"):
    model.addFrame(
      pin.Frame(name, parent, placement(chain[name]), pin.FrameType.OP_FRAME)
    )
  model.lowerPositionLimit = np.array(chain["lower"])
  model.upperPositionLimit = np.array(chain["upper"])
  model.velocityLimit = np.array(chain["vmax"])
  config = pink.Configuration(model, model.createData(), np.array(chain["home"]))
  frame = pink.tasks.FrameTask(
    "tag",
    position_cost=settings.position_cost,
    orientation_cost=settings.orientation_cost,
    lm_damping=settings.lm_damping,
  )
  posture = pink.tasks.PostureTask(cost=settings.posture_cost)
  posture.set_target_from_configuration(config)
  tasks = [frame, posture, pink.tasks.DampingTask(cost=settings.damping_cost)]
  qp_solver = "daqp"
  reply(
    joints=[j["name"] for j in chain["joints"]],
    posture_reference_rad=chain["home"],
    settings=asdict(settings),
    pink=pink.__version__,
    qpsolvers=version("qpsolvers"),
    daqp=version(qp_solver),
    pinocchio=pin.__version__,
    python=sys.version.split()[0],
    first_frame="scipy bounded pose IK from home",
    qp_solver=qp_solver,
  )
  for line in sys.stdin:
    request = json.loads(line)
    if "fk" in request:
      config.update(np.array(request["fk"]))
      reply(
        **{
          name: config.get_transform_frame_to_world(name).homogeneous.tolist()
          for name in ("tag", "flange")
        }
      )
      continue
    config.update(np.array(request["previous"]))
    frame.set_target(placement(request["target"]))
    dt = request["dt"]
    try:
      v = pink.solve_ik(config, tasks, dt, solver=qp_solver)
    except pink.exceptions.NoSolutionFound as exc:
      reply(q=None, status=str(exc), projected=[])
      continue
    reply(q=config.integrate(v, dt).tolist(), status="QP solved", projected=[])


if __name__ == "__main__":
  main()
