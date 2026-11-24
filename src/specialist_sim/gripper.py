"""The described gripper: a base, two fingers on one servo, their pads, and the
lens on the base. A hand's state is the pose of its base tag, the dataset's hand
frame, and its opening, the right finger's angle negated. The fingers are
coupled, so this differs negligibly from the dataset's ``(left - right) / 2``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import specialist_description

from specialist_sim.arrays import Float64Array, Int64Array
from specialist_sim.camera.lens import Lens

# Every body a hand owns carries ``{hand}_gripper_`` as a prefix.
HANDS = ("left", "right")
# The side of the room along y each hand works from, for an operator at -x
# facing +x.
ACROSS = {"left": 1.0, "right": -1.0}

# The opening when the pads meet, and at its open limit.
SHUT = -0.1518
OPEN = 0.6981

# The geom group the lens tracer and the overview video draw.
DRAWN = 1
# The contact dimensionality the description marks a gripping pad with.
GRIPPING = 4

# The described parts a lens looks out of, and so cannot see.
HOUSING = ("x3_", "protector")
BASE_BODY = "base"
TAG_BODY = "base_tag"


@dataclass(frozen=True)
class Hand:
  name: str
  base: int
  """The described base, which carries the lens site."""
  tag: int
  """The base tag body, whose pose is the hand's state."""
  lens_site: int
  fingers: tuple[int, int]
  """Where the left and right finger angles sit in ``qpos``."""
  servo: int
  """The right finger's position servo, which the left finger follows."""
  pads: tuple[int, ...]
  """Geom ids."""
  housing: tuple[int, ...]
  """The geoms of the ``HOUSING`` its lens looks out of."""


def _described(
  side: int, finger: tuple[float, float], model_path: Path | None
) -> mujoco.MjSpec:
  """The gripper as described, with a finger servo of ``finger``'s stiffness and
  damping. The hand at ``side``, its index in ``HANDS``, gets a contact bit of
  its own that meets the other's, since the description lets two grippers pass
  through each other."""
  asset = (
    specialist_description.get_mjcf_path("specialist_actuated")
    if model_path is None
    else model_path
  )
  described = mujoco.MjSpec.from_file(str(asset))
  # A servo on the right finger; the description couples the left to it.
  servo = described.add_actuator(
    name="right_finger", target="right_finger", trntype=mujoco.mjtTrn.mjTRN_JOINT
  )
  servo.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  stiffness, damping = finger
  servo.gainprm[0] = stiffness
  servo.biasprm[1] = -stiffness
  servo.biasprm[2] = -damping
  for geom in described.geoms:
    if geom.contype == 2:
      geom.contype = 2 << side
      geom.conaffinity = 1 | (2 << (1 - side))
  return described


def attach(
  spec: mujoco.MjSpec,
  body: mujoco.MjsBody,
  hand: str,
  lens: Lens,
  finger: tuple[float, float],
  *,
  model_path: Path | None = None,
  mount_body: str = TAG_BODY,
  base_from_tag: Float64Array | None = None,
) -> None:
  """Align the named gripper body with the carrier; default to the stock tag."""
  described = _described(HANDS.index(hand), finger, model_path)
  base = described.body(BASE_BODY)
  if base is None:
    raise ValueError(f"gripper has no {BASE_BODY} body")
  if mount_body == BASE_BODY and (
    base.parent.name != described.worldbody.name or base.joints
  ):
    raise ValueError(f"gripper {BASE_BODY} must be fixed and directly under worldbody")
  tag = described.body(TAG_BODY)
  if base_from_tag is not None:
    if tag is not None:
      raise ValueError(
        f"gripper already defines {TAG_BODY}; omit gripper_base_from_tag"
      )
    base.add_body(name=TAG_BODY, pos=base_from_tag[:3], quat=base_from_tag[3:])
  elif tag is None:
    raise ValueError(f"gripper has no {TAG_BODY}; supply gripper_base_from_tag")
  anchor = described.body(mount_body)
  if anchor is None:
    raise ValueError(f"gripper has no mount body {mount_body}")
  pos, quat = np.zeros(3), np.zeros(4)
  mujoco.mju_negPose(pos, quat, anchor.pos, anchor.quat)
  spec.attach(
    described, prefix=f"{hand}_gripper_", frame=body.add_frame(pos=pos, quat=quat)
  )
  spec.body(f"{hand}_gripper_{BASE_BODY}").add_site(
    name=f"{hand}_gripper_lens", pos=list(lens.pos), quat=list(lens.quat)
  )


def bind(model: mujoco.MjModel, hand: str) -> Hand:
  prefix = f"{hand}_gripper_"
  left, right = (model.joint(f"{prefix}{side}_finger") for side in ("left", "right"))
  fingers = [model.body(f"{prefix}{side}_finger").id for side in ("left", "right")]
  pads = (model.geom_condim == GRIPPING) & np.isin(model.geom_bodyid, fingers)
  housing = tuple(f"{prefix}{part}" for part in HOUSING)
  return Hand(
    name=hand,
    base=model.body(f"{prefix}{BASE_BODY}").id,
    tag=model.body(f"{prefix}{TAG_BODY}").id,
    lens_site=model.site(f"{prefix}lens").id,
    fingers=(int(left.qposadr[0]), int(right.qposadr[0])),
    servo=int(model.actuator(right.name).id),
    pads=tuple(np.flatnonzero(pads).tolist()),
    housing=tuple(
      geom for geom in range(model.ngeom) if model.geom(geom).name.startswith(housing)
    ),
  )


def grip(ctrl: Float64Array, servos: Int64Array, opening: Float64Array) -> None:
  """Send every hand's servo, ``(hands,)``, an opening, ``(num_envs, hands)``:
  the servo turns the right finger, which stands at minus the opening."""
  ctrl[:, servos] = -opening


def states(
  hands: Sequence[Hand],
  qpos: Float64Array,
  xpos: Float64Array,
  xquat: Float64Array,
) -> Float64Array:
  """Every hand read out of ``n`` kinematics, ``qpos`` ``(n, nq)``, ``xpos``
  ``(n, nbody, 3)`` and ``xquat`` ``(n, nbody, 4)``: its base tag's position
  and quaternion, then its opening, ``(n, hands, 8)``."""
  tag = [hand.tag for hand in hands]
  opening = -qpos[:, [hand.fingers[1] for hand in hands], None]
  return np.concatenate([xpos[:, tag], xquat[:, tag], opening], axis=-1)
