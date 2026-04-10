"""Latched subtask bonuses + bounded continuous distance bridge for BEHAVIOR PPO.

Implements the classical "staged dense reward" pattern (reach → grasp →
lift → place from robosuite Pick-and-Place) but adapted to BEHAVIOR-1K
tasks. Each task's BDDL goal is decomposed into subtasks of two kinds:

  1. **Latched bonuses**: a bonus emitted ONCE per episode the first
     time the corresponding boolean predicate becomes True. Cannot be
     farmed (no marginal reward after the first crossing). Examples for
     a (toggled_on radio) goal: `near_radio` (0.3), `touching_radio` (0.5).

  2. **Continuous delta bridges** (NEW in iter 3-B): per-step
     potential-style distance shaping that pulls the EEF toward a target
     object until the corresponding `near_X` latched bonus fires, then
     gates off. Form: `(prev_dist - current_dist) * coeff`, telescoping
     over the episode to `(initial_dist - final_dist) * coeff`. Bounded
     per-episode total ≈ initial_dist × coeff regardless of episode
     length, so the shaping cannot dominate the terminal PotentialReward
     via accumulation. Hovering produces zero net delta.

     Why we need both kinds: iter 2.5 with only latched binary bonuses
     produced reward_nonzero_frac=0 across 5 PPO steps. The random-init
     R1Pro starts ~2 m from the target object, and a 0.25 m near
     threshold is unreachable from random exploration in 500-step
     episodes. The continuous bridge provides gradient *before* the
     threshold is crossed, so the policy has something to descend.

     Why this isn't the same imitation problem we had with
     DemoEEFDistanceReward: the target is the BDDL goal's actual
     argument object (taken from the task's BDDL `(:goal ...)` block),
     not a fixed expert trajectory. Multiple valid policies can satisfy
     "approach the radio". The shaping just says "go toward this thing".

Architecture: BDDL-predicate templates + per-task overrides
    Most BEHAVIOR-1K tasks have flat `(and ...)` goal conjunctions of
    well-known BDDL predicates (`inside`, `ontop`, `nextto`, `under`,
    `touching`, `attached`, `toggled_on`, `open`, etc.). For each
    templatable predicate we register a function that auto-derives the
    subtask list (latched + continuous) from the goal predicate's args.

    Tasks with non-templatable predicates (`covered`, `cooked`,
    `on_fire`, `real`, `contains`, `filled`) — typically cleaning and
    cooking tasks where the relevant action involves a tool/appliance
    not named in the goal — fall through to TASK_SUBTASKS_OVERRIDE for
    hand-crafted decompositions, or get no shaping at all (the task
    still has its terminal PotentialReward).

This reward composes additively with the existing PotentialReward
(which handles the BDDL goal predicate via potential delta) — subtask
bonuses + continuous bridges are intermediate stepping stones, not a
replacement for the goal signal.
"""

from typing import Any, Callable, Optional

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.object_states import IsGrasping, Pose, Touching
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger("BehaviorSubtaskReward")


# ---------------------------------------------------------------------------
# Predicate helper primitives. Each takes (task, env, ...) and returns either
# a bool (for latched predicates) or a float (for continuous distance).
# Defensive — return False / None on any lookup error so a missing object
# never crashes the reward.
# ---------------------------------------------------------------------------

# Object-name shortener for log/info field readability — strips the BDDL
# noun-sense suffix (".n.01_1") down to a human label (e.g. "radio_receiver").
def _short(name: str) -> str:
    base = name.split(".n.")[0] if ".n." in name else name
    return base.rstrip("_")


def _eef_to_obj_min_distance(task, env, obj_scope_name: str) -> Optional[float]:
    """Return min L2 distance from any of the robot's end-effectors to the
    BDDL-bound object's position. None on lookup error."""
    try:
        obj = task.object_scope[obj_scope_name].wrapped_obj
        obj_pos, _ = obj.states[Pose].get_value()
    except Exception:
        return None
    robot = env.robots[0]
    arm_names = list(robot.arm_names) if hasattr(robot, "arm_names") else [robot.default_arm]
    obj_t = th.as_tensor(obj_pos, dtype=th.float32)
    best: Optional[float] = None
    for arm in arm_names[:2]:
        try:
            eef = robot.get_eef_position(arm)
        except Exception:
            continue
        d = T.l2_distance(th.as_tensor(eef, dtype=th.float32), obj_t).item()
        if best is None or d < best:
            best = d
    return best


def _eef_within(task, env, obj_scope_name: str, threshold: float) -> bool:
    """True if the closest EEF is within `threshold` meters of the
    BDDL-bound object."""
    d = _eef_to_obj_min_distance(task, env, obj_scope_name)
    return d is not None and d < threshold


def _robot_touching(task, env, obj_scope_name: str) -> bool:
    """True if any rigid body of the robot is in contact with the
    BDDL-bound object. Uses OmniGibson's Touching state internally."""
    try:
        obj = task.object_scope[obj_scope_name].wrapped_obj
        return bool(obj.states[Touching].get_value(env.robots[0]))
    except Exception:
        return False


def _robot_grasping(task, env, obj_scope_name: str) -> bool:
    """True if the robot is currently grasping the BDDL-bound object via
    any arm. Uses OmniGibson's IsGrasping robot state."""
    try:
        obj = task.object_scope[obj_scope_name].wrapped_obj
        return bool(env.robots[0].states[IsGrasping].get_value(obj))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Subtask schema
#
# Each subtask is a dict with at minimum a "name" and a "kind" field. Two
# kinds supported:
#
#   {"name": str, "kind": "latched",
#    "fn": Callable[[task, env], bool], "bonus": float}
#       Fires `bonus * scale` once per episode the first time `fn` is True.
#       Subsequent steps emit 0 until reset.
#
#   {"name": str, "kind": "continuous_delta",
#    "dist_fn": Callable[[task, env], Optional[float]],
#    "coeff": float, "gate_off_when": str}
#       Per-step reward = `(prev_dist - current_dist) * coeff * scale`.
#       Telescopes to bounded total over the episode.
#       Becomes inactive once the latched subtask named in `gate_off_when`
#       has fired (the milestone is passed; further bridge would be noise).
#
# Templates produce a mix of both. The resolver returns a flat list of
# subtask dicts; BehaviorSubtaskReward._step iterates and dispatches by kind.
# ---------------------------------------------------------------------------

# meters; EEF-to-object distance for "near_X" latched predicates.
# Tight enough that "near" actually means "EEF in striking distance",
# loose enough that the policy can achieve it once the continuous bridge
# pulls the EEF into the right region.
NEAR_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# BDDL-predicate templates. Each takes a list of object args and returns
# a flat list of subtask dicts (mix of latched + continuous_delta).
#
# Bonus magnitudes are chosen so the total per-episode shaping is in the
# same band as the terminal PotentialReward (which fires +1.0 on the BDDL
# goal flip):
#   continuous bridge per target:  initial_dist × coeff ≈ 1.0 × 0.5 = 0.5
#   latched near_X:                                              0.2-0.3
#   latched grasping_X / touching_X:                             0.4-0.5
#   terminal PotentialReward:                                    1.0
# ---------------------------------------------------------------------------


def _template_unary_state(args: list[str]) -> list[dict[str, Any]]:
    """For (toggled_on X), (open X) — single-object state changes.

    Decomposition: continuous bridge toward X, latched near_X, latched
    touching_X. The actual state change (toggle/open) is left to
    PotentialReward via the BDDL goal evaluation.
    """
    obj = args[0]
    short = _short(obj)
    return [
        {"name": f"delta_to_{short}", "kind": "continuous_delta",
         "dist_fn": lambda t, e, o=obj: _eef_to_obj_min_distance(t, e, o),
         "coeff": 0.5, "gate_off_when": f"near_{short}"},
        {"name": f"near_{short}", "kind": "latched",
         "fn": lambda t, e, o=obj: _eef_within(t, e, o, NEAR_THRESHOLD),
         "bonus": 0.3},
        {"name": f"touching_{short}", "kind": "latched",
         "fn": lambda t, e, o=obj: _robot_touching(t, e, o),
         "bonus": 0.5},
    ]


def _template_binary_placement(args: list[str]) -> list[dict[str, Any]]:
    """For (inside X Y), (ontop X Y), (under X Y), (nextto X Y),
    (touching X Y), (attached X Y) — placement of X relative to Y.

    Decomposition: continuous bridges toward both X and Y (independently
    gated), latched near_X, latched grasping_X, latched near_Y. The
    actual relational predicate is left to PotentialReward.
    """
    obj, target = args[0], args[1]
    obj_short, target_short = _short(obj), _short(target)
    return [
        {"name": f"delta_to_{obj_short}", "kind": "continuous_delta",
         "dist_fn": lambda t, e, o=obj: _eef_to_obj_min_distance(t, e, o),
         "coeff": 0.3, "gate_off_when": f"near_{obj_short}"},
        {"name": f"delta_to_{target_short}", "kind": "continuous_delta",
         "dist_fn": lambda t, e, o=target: _eef_to_obj_min_distance(t, e, o),
         "coeff": 0.3, "gate_off_when": f"near_{target_short}"},
        {"name": f"near_{obj_short}", "kind": "latched",
         "fn": lambda t, e, o=obj: _eef_within(t, e, o, NEAR_THRESHOLD),
         "bonus": 0.2},
        {"name": f"grasping_{obj_short}", "kind": "latched",
         "fn": lambda t, e, o=obj: _robot_grasping(t, e, o),
         "bonus": 0.4},
        {"name": f"near_{target_short}", "kind": "latched",
         "fn": lambda t, e, o=target: _eef_within(t, e, o, NEAR_THRESHOLD + 0.05),
         "bonus": 0.3},
    ]


# Map BDDL predicate name -> template function. Templatable predicates only.
GOAL_PREDICATE_TEMPLATES: dict[str, Callable[[list[str]], list[dict[str, Any]]]] = {
    # Unary state changes
    "toggled_on": _template_unary_state,
    "open":       _template_unary_state,
    # Binary placements / spatial relations
    "inside":     _template_binary_placement,
    "ontop":      _template_binary_placement,
    "under":      _template_binary_placement,
    "nextto":     _template_binary_placement,
    "touching":   _template_binary_placement,
    "attached":   _template_binary_placement,
    # Non-templatable (need overrides or accept no shaping):
    #   covered, contains, filled — substance-based, no source object in goal
    #   cooked, on_fire, real     — produced by appliance/transformation
}


# ---------------------------------------------------------------------------
# Per-task BDDL goal predicates, extracted from
# BEHAVIOR-1K/bddl3/bddl/activity_definitions/<task>/problem0.bddl
# ---------------------------------------------------------------------------

TASK_GOAL_PREDICATES: dict[str, list[tuple[str, list[str]]]] = {
    # 0
    "turning_on_radio": [("toggled_on", ["radio_receiver.n.01_1"])],
    # 1
    "picking_up_trash": [("inside", ["can__of__soda.n.01_1", "ashcan.n.01_1"])],
    # 2
    "putting_away_Halloween_decorations": [("inside", ["pumpkin.n.02_1", "cabinet.n.01_1"])],
    # 3
    "cleaning_up_plates_and_food": [("ontop", ["pizza.n.01_1", "plate.n.04_1"])],
    # 4
    "can_meat": [("inside", ["hinged_jar.n.01_1", "cabinet.n.01_1"])],
    # 5
    "setting_mousetraps": [("ontop", ["mousetrap.n.01_1", "floor.n.01_1"])],
    # 6
    "hiding_Easter_eggs": [("nextto", ["easter_egg.n.01_1", "tree.n.01_1"])],
    # 7
    "picking_up_toys": [("inside", ["jigsaw_puzzle.n.01_1", "toy_box.n.01_1"])],
    # 8
    "rearranging_kitchen_furniture": [
        ("inside", ["toaster.n.02_1", "cabinet.n.01_1"]),
        ("inside", ["food_processor.n.01_1", "cabinet.n.01_1"]),
        ("inside", ["french_press.n.01_1", "cabinet.n.01_1"]),
        ("open",   ["cabinet.n.01_1"]),
    ],
    # 9
    "putting_up_Christmas_decorations_inside": [
        ("nextto",   ["gift_box.n.01_1", "christmas_tree.n.05_1"]),
        ("under",    ["gift_box.n.01_1", "christmas_tree.n.05_1"]),
        ("touching", ["gift_box.n.01_1", "christmas_tree.n.05_1"]),
    ],
    # 10
    "set_up_a_coffee_station_in_your_kitchen": [
        ("ontop",  ["coffee_maker.n.01_1", "countertop.n.01_1"]),
        ("nextto", ["bottle__of__coffee.n.01_1", "coffee_maker.n.01_1"]),
        ("ontop",  ["paper_coffee_filter.n.01_1", "coffee_maker.n.01_1"]),
        ("nextto", ["saucer.n.02_1", "coffee_maker.n.01_1"]),
        ("ontop",  ["coffee_cup.n.01_1", "saucer.n.02_1"]),
        ("nextto", ["electric_kettle.n.01_1", "coffee_maker.n.01_1"]),
    ],
    # 11
    "putting_dishes_away_after_cleaning": [("inside", ["plate.n.04_1", "cabinet.n.01_1"])],
    # 12
    "preparing_lunch_box": [("inside", ["half__apple.n.01_1", "packing_box.n.02_1"])],
    # 13
    "loading_the_car": [
        ("inside", ["container.n.01_1", "car.n.01_1"]),
        ("inside", ["digital_camera.n.01_1", "container.n.01_1"]),
        ("inside", ["tennis_racket.n.01_1", "car.n.01_1"]),
    ],
    # 14
    "carrying_in_groceries": [
        ("inside", ["beefsteak_tomato.n.01_1", "electric_refrigerator.n.01_1"]),
        ("inside", ["carton__of__milk.n.01_1", "electric_refrigerator.n.01_1"]),
    ],
    # 15
    "bringing_in_wood": [("ontop", ["plywood.n.01_1", "floor.n.01_2"])],
    # 16
    "moving_boxes_to_storage": [
        ("ontop", ["storage_container.n.01_1", "floor.n.01_2"]),
        ("ontop", ["storage_container.n.01_2", "storage_container.n.01_1"]),
    ],
    # 17
    "bringing_water": [("ontop", ["bottle.n.01_1", "coffee_table.n.01_1"])],
    # 18
    "tidying_bedroom": [
        ("nextto", ["sandal.n.01_1", "bed.n.01_1"]),
        ("nextto", ["sandal.n.01_2", "sandal.n.01_1"]),
        ("ontop",  ["book.n.02_1", "table.n.02_1"]),
    ],
    # 19
    "outfit_a_basic_toolbox": [
        ("inside", ["drill.n.01_1", "toolbox.n.01_1"]),
        ("inside", ["pliers.n.01_1", "toolbox.n.01_1"]),
        ("inside", ["flashlight.n.01_1", "toolbox.n.01_1"]),
        ("inside", ["allen_wrench.n.01_1", "toolbox.n.01_1"]),
        ("inside", ["screwdriver.n.01_1", "toolbox.n.01_1"]),
        ("ontop",  ["toolbox.n.01_1", "tabletop.n.01_1"]),
        ("open",   ["toolbox.n.01_1"]),
    ],
    # 20
    "sorting_vegetables": [("inside", ["bok_choy.n.02_1", "mixing_bowl.n.01_1"])],
    # 21
    "collecting_childrens_toys": [("inside", ["die.n.01_1", "bookcase.n.01_1"])],
    # 22
    "putting_shoes_on_rack": [
        ("touching", ["sandal.n.01_1", "hallstand.n.01_1"]),
        ("touching", ["sandal.n.01_1", "floor.n.01_1"]),
    ],
    # 23
    "boxing_books_up_for_storage": [("inside", ["book.n.02_1", "box.n.01_1"])],
    # 24
    "storing_food": [("inside", ["box__of__oatmeal.n.01_1", "cabinet.n.01_1"])],
    # 25
    "clearing_food_from_table_into_fridge": [("inside", ["half__apple_pie.n.01_1", "tupperware.n.01_1"])],
    # 26
    "assembling_gift_baskets": [("inside", ["candle.n.01_1", "wicker_basket.n.01_1"])],
    # 27
    "sorting_household_items": [("under", ["bottle__of__detergent.n.01_1", "sink.n.01_1"])],
    # 28
    "getting_organized_for_work": [
        ("nextto", ["keyboard.n.01_1", "monitor.n.04_1"]),
        ("ontop",  ["keyboard.n.01_1", "desk.n.01_1"]),
        ("under",  ["computer.n.01_1", "desk.n.01_1"]),
        ("ontop",  ["monitor.n.04_1", "desk.n.01_1"]),
        ("nextto", ["mouse.n.04_1", "keyboard.n.01_1"]),
        ("ontop",  ["mouse.n.04_1", "desk.n.01_1"]),
        ("nextto", ["folder.n.02_1", "mouse.n.04_1"]),
        ("ontop",  ["notebook.n.01_1", "folder.n.02_1"]),
        ("ontop",  ["pen.n.01_1", "notebook.n.01_1"]),
        ("nextto", ["swivel_chair.n.01_1", "desk.n.01_1"]),
    ],
    # 29
    "clean_up_your_desk": [("inside", ["folder.n.02_1", "bookcase.n.01_1"])],
    # 30 — partial: ontop+inside templated, on_fire ignored
    "setting_the_fire": [
        ("ontop",   ["firewood.n.01_1", "newspaper.n.03_1"]),
        ("inside",  ["firewood.n.01_1", "wood_fireplace.n.01_1"]),
    ],
    # 31..33: only `covered` predicates → no template coverage
    "clean_boxing_gloves": [],
    "wash_a_baseball_cap": [],
    "wash_dog_toys":       [],
    # 34
    "hanging_pictures": [("attached", ["poster.n.01_1", "wall_nail.n.01_1"])],
    # 35
    "attach_a_camera_to_a_tripod": [("attached", ["digital_camera.n.01_1", "camera_tripod.n.01_1"])],
    # 36..39: only `covered` (and 38 is forall) → no template coverage
    "clean_a_patio":       [],
    "clean_a_trumpet":     [],
    "spraying_for_bugs":   [],
    "spraying_fruit_trees": [],
    # 40..41: only real+contains → no template coverage
    "make_microwave_popcorn": [],
    "cook_cabbage":           [],
    # 42 — partial: inside templates apply, real/contains ignored
    "chop_an_onion": [
        ("inside", ["parer.n.02_1", "sink.n.01_1"]),
        ("inside", ["chopping_board.n.01_1", "sink.n.01_1"]),
    ],
    # 43 — partial: open templates apply, real ignored
    "slicing_vegetables": [
        ("open", ["electric_refrigerator.n.01_1"]),
    ],
    # 44 — only real → no template coverage
    "chopping_wood": [],
    # 45..46: only cooked → no template coverage
    "cook_hot_dogs": [],
    "cook_bacon":    [],
    # 47
    "freeze_pies": [("inside", ["apple_pie.n.01_1", "tupperware.n.01_1"])],
    # 48 — only real+filled+contains → no template coverage
    "canning_food": [],
    # 49 — partial: ontop applies, real ignored
    "make_pizza": [
        ("ontop", ["pizza.n.01_1", "cookie_sheet.n.01_1"]),
    ],
}


# ---------------------------------------------------------------------------
# Per-task hand-crafted overrides for tasks where the auto-template is
# wrong/insufficient. Format: activity_name -> list of subtask dicts (same
# schema as template output). Currently empty — adding entries here is the
# growth path for the 13 tasks with cooking/cleaning predicates that don't
# fit the templates.
# ---------------------------------------------------------------------------

TASK_SUBTASKS_OVERRIDE: dict[str, list[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Resolver: activity_name → list of subtask dicts.
# Priority: TASK_SUBTASKS_OVERRIDE > template-derived > empty.
# ---------------------------------------------------------------------------

def get_subtasks_for_task(activity_name: str) -> list[dict[str, Any]]:
    """Return the subtask list for a given BEHAVIOR activity.

    Empty list means "no shaping for this task" — the BehaviorSubtaskReward
    will return 0.0 every step, leaving only the existing PotentialReward
    to drive learning.
    """
    if activity_name in TASK_SUBTASKS_OVERRIDE:
        return TASK_SUBTASKS_OVERRIDE[activity_name]

    goal_predicates = TASK_GOAL_PREDICATES.get(activity_name, [])
    if not goal_predicates:
        return []

    subtasks: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for pred_name, args in goal_predicates:
        template = GOAL_PREDICATE_TEMPLATES.get(pred_name)
        if template is None:
            continue
        for entry in template(args):
            name = entry["name"]
            if name in seen_names:
                # De-duplicate across goal predicates that touch the same
                # object (e.g. (ontop X Y) and (touching X Z) both emit
                # near_X / grasping_X / delta_to_X). First occurrence wins.
                continue
            seen_names.add(name)
            subtasks.append(entry)
    return subtasks


# ---------------------------------------------------------------------------
# Reward function class
# ---------------------------------------------------------------------------

class BehaviorSubtaskReward(BaseRewardFunction):
    """Latched per-subtask bonuses + continuous distance bridges for a
    BEHAVIOR-1K activity.

    On first `_step` call, looks up the subtask list for the task's
    activity_name via `get_subtasks_for_task`. Then on each step:
      - Latched subtasks: emit `bonus * scale` once on the first True
        transition; subsequent steps emit 0 until reset.
      - Continuous_delta subtasks: emit `(prev_dist - cur_dist) * coeff
        * scale` per step until the gating latched subtask fires, then
        contribute 0.

    If the activity has no registered subtasks (and no override), the
    reward is a no-op (returns 0.0 every step) — safe for tasks that
    haven't been decomposed yet.

    Args:
        scale: global multiplier on all subtask reward components. The
            per-subtask bonus values and coefficients in the templates
            are the "natural" magnitudes; this lets users dial total
            shaping intensity up or down via the YAML `r_subtask_bonus`
            config without editing code.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self._scale = float(scale)
        self._subtasks: list[dict[str, Any]] = []
        self._fired: dict[str, bool] = {}
        self._prev_dists: dict[str, float] = {}
        self._activity_name: Optional[str] = None

    def _ensure_loaded(self, task) -> None:
        if self._activity_name is not None:
            return
        self._activity_name = getattr(task, "activity_name", None) or "<unknown>"
        self._subtasks = get_subtasks_for_task(self._activity_name)
        self._fired = {s["name"]: False for s in self._subtasks if s["kind"] == "latched"}
        self._prev_dists = {}
        if self._subtasks:
            counts = {"latched": 0, "continuous_delta": 0}
            names = []
            for s in self._subtasks:
                counts[s["kind"]] = counts.get(s["kind"], 0) + 1
                names.append(s["name"])
            logger.info(
                f"BehaviorSubtaskReward enabled for activity '{self._activity_name}': "
                f"{counts['latched']} latched + {counts.get('continuous_delta', 0)} continuous "
                f"({', '.join(names)}), scale={self._scale}"
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
        info: dict[str, Any] = {}

        for subtask in self._subtasks:
            kind = subtask["kind"]
            name = subtask["name"]

            if kind == "latched":
                if self._fired.get(name, False):
                    info[f"subtask_{name}"] = 1
                    continue
                try:
                    achieved = bool(subtask["fn"](task, env))
                except Exception:
                    achieved = False
                if achieved:
                    self._fired[name] = True
                    total += subtask["bonus"] * self._scale
                    info[f"subtask_{name}"] = 1
                else:
                    info[f"subtask_{name}"] = 0

            elif kind == "continuous_delta":
                gate = subtask.get("gate_off_when")
                if gate and self._fired.get(gate, False):
                    # Milestone passed; bridge no longer active.
                    continue
                try:
                    cur = subtask["dist_fn"](task, env)
                except Exception:
                    cur = None
                if cur is None:
                    continue
                prev = self._prev_dists.get(name)
                if prev is not None:
                    delta_reward = (prev - cur) * subtask["coeff"] * self._scale
                    total += delta_reward
                self._prev_dists[name] = cur

        return total, info

    def reset(self, task, env):
        super().reset(task, env)
        self._fired = {name: False for name in self._fired}
        self._prev_dists = {}
