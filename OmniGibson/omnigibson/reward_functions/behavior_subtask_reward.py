"""One-shot latched per-subtask bonuses for BEHAVIOR PPO bootstrapping.

Implements the classical "staged dense reward" pattern (reach → grasp →
lift → place from robosuite Pick-and-Place) but adapted to BEHAVIOR-1K
tasks. Each subtask is a boolean predicate evaluated per env step from
world state. The first time a predicate becomes True within an episode,
the corresponding bonus is added once. Subsequent steps within the same
episode never re-emit the same bonus (latched). On episode reset, all
latch flags clear.

Why latched + boolean:
    A bonus that fires once and cannot be farmed avoids the reward-hacking
    failure mode of continuous distance shaping (e.g. policy hovers near
    target object accumulating delta-distance reward without ever
    completing the task).

Why per-task predicate dict:
    Predicates need access to runtime objects (task.object_scope, env.robots,
    object states) so they can't be serialized in YAML. We register them in a
    `TASK_SUBTASKS` dict keyed by activity_name. Adding a new task means
    adding one entry to this dict.

This reward composes additively with the existing PotentialReward (which
handles the BDDL goal predicate via potential delta) — subtask bonuses are
intermediate stepping stones, not a replacement for the goal signal.
"""

from typing import Callable

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.object_states import Pose, Touching
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger("BehaviorSubtaskReward")


# ---------------------------------------------------------------------------
# Per-task predicate definitions.
# Each entry is (subtask_name, predicate_fn, bonus_value).
# predicate_fn signature: (task, env) -> bool
#   - task: BehaviorTask instance (has object_scope, activity_name, ...)
#   - env: omnigibson Environment (has robots[])
# Add a new task by registering its predicates here.
# ---------------------------------------------------------------------------


def _eef_within(task, env, obj_scope_name: str, threshold: float) -> bool:
    """True if either of the robot's end-effectors is within `threshold`
    meters of the BDDL-bound object's position. Returns False on any
    lookup error so a missing object never crashes the reward."""
    try:
        obj = task.object_scope[obj_scope_name].wrapped_obj
        obj_pos, _ = obj.states[Pose].get_value()
    except Exception:
        return False
    robot = env.robots[0]
    arm_names = list(robot.arm_names) if hasattr(robot, "arm_names") else [robot.default_arm]
    for arm in arm_names[:2]:
        try:
            eef = robot.get_eef_position(arm)
        except Exception:
            continue
        if T.l2_distance(th.as_tensor(eef, dtype=th.float32),
                          th.as_tensor(obj_pos, dtype=th.float32)).item() < threshold:
            return True
    return False


def _robot_touching(task, env, obj_scope_name: str) -> bool:
    """True if any rigid body of the robot is in contact with the
    BDDL-bound object. Uses OmniGibson's Touching state which internally
    checks ContactBodies set intersection."""
    try:
        obj = task.object_scope[obj_scope_name].wrapped_obj
        return bool(obj.states[Touching].get_value(env.robots[0]))
    except Exception:
        return False


# Subtask registry: activity_name -> list of (name, fn, bonus)
TASK_SUBTASKS: dict[str, list[tuple[str, Callable, float]]] = {
    "turning_on_radio": [
        # Stage 1: get either EEF within 25 cm of the radio.
        (
            "near_radio",
            lambda task, env: _eef_within(task, env, "radio_receiver.n.01_1", 0.25),
            0.5,
        ),
        # Stage 2: any robot link in physical contact with the radio.
        # The actual toggle event (radio_on) is intentionally NOT a
        # subtask — it's already covered by PotentialReward via the BDDL
        # (toggled_on radio_receiver.n.01_1) goal predicate, and
        # double-counting would distort the reward landscape.
        (
            "touching_radio",
            lambda task, env: _robot_touching(task, env, "radio_receiver.n.01_1"),
            1.0,
        ),
    ],
}


class BehaviorSubtaskReward(BaseRewardFunction):
    """Latched per-subtask bonuses for a BEHAVIOR-1K activity.

    Looks up the per-task predicate list in TASK_SUBTASKS by
    `task.activity_name`. If the activity has no entry, the reward is a
    no-op (returns 0.0 every step) — safe for tasks that haven't been
    decomposed yet.

    Args:
        scale: global multiplier on all subtask bonuses. The per-subtask
            bonus values in TASK_SUBTASKS are the "natural" magnitudes;
            this lets users dial total shaping intensity up or down without
            editing the registry.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self._scale = float(scale)
        self._subtasks: list[tuple[str, Callable, float]] = []
        self._fired: dict[str, bool] = {}
        self._activity_name: str | None = None

    def _ensure_loaded(self, task) -> None:
        """Lazy lookup of the predicate list — task may not be fully
        constructed at __init__ time, so we resolve on the first _step."""
        if self._activity_name is not None:
            return
        self._activity_name = getattr(task, "activity_name", None) or "<unknown>"
        self._subtasks = TASK_SUBTASKS.get(self._activity_name, [])
        self._fired = {name: False for name, _, _ in self._subtasks}
        if self._subtasks:
            names = ", ".join(name for name, _, _ in self._subtasks)
            logger.info(
                f"BehaviorSubtaskReward enabled for activity '{self._activity_name}': "
                f"{len(self._subtasks)} subtasks ({names}), scale={self._scale}"
            )
        else:
            logger.info(
                f"BehaviorSubtaskReward: no subtasks registered for activity "
                f"'{self._activity_name}', reward will be no-op"
            )

    def _step(self, task, env, action):
        self._ensure_loaded(task)
        if not self._subtasks:
            return 0.0, {}

        total = 0.0
        info: dict = {}
        for name, fn, bonus in self._subtasks:
            if self._fired[name]:
                info[f"subtask_{name}"] = 1
                continue
            try:
                achieved = bool(fn(task, env))
            except Exception:
                achieved = False
            if achieved:
                self._fired[name] = True
                total += bonus * self._scale
                info[f"subtask_{name}"] = 1
            else:
                info[f"subtask_{name}"] = 0
        return total, info

    def reset(self, task, env):
        super().reset(task, env)
        # Re-resolve activity_name lazily on next _step in case it changed,
        # but clear all latches now.
        self._fired = {name: False for name in self._fired}
