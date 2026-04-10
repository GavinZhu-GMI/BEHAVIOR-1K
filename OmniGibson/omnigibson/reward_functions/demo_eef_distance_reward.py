"""Demo-pose end-effector distance shaping reward.

A potential-based shaping reward that pulls the robot's end-effectors
toward an expert demonstration trajectory's EEF positions at the
matching timestep. Designed for BEHAVIOR-1K tasks where the BDDL goal
predicates are too coarse to provide a useful gradient for a
randomly-initialized RL policy.

Reference: TGRPO (Trajectory-wise GRPO for VLA Fine-tuning,
https://arxiv.org/html/2506.08440), section 3.2 — the `f2` term uses
Euclidean distance from current end-effector to demonstration
end-effector at the matching task-progress index. The delta
formulation here makes the per-episode total telescope to
`(initial_dist - final_dist) * dist_coeff`, bounded regardless of
episode length, so the shaping cannot dominate any terminal task
reward via accumulation.

The demo file is a (T, 6) numpy array of per-frame
`[eef_left_xyz, eef_right_xyz]` extracted from a successful expert
trajectory. For BEHAVIOR-1K specifically the file is generated from
the LeRobot dataset `behavior-1k/2025-challenge-demos` by reading
`observation.state[:, 186:189]` and `[:, 225:228]`.
"""

import os
from pathlib import Path

import numpy as np
import torch as th

from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger("DemoEEFDistanceReward")

_DEMO_DATA_DIR = Path(__file__).resolve().parent / "demo_data"


class DemoEEFDistanceReward(BaseRewardFunction):
    """Per-step delta-distance shaping vs an expert-demo EEF trajectory.

    On reset, resets the per-frame pointer to 0. On each `_step`,
    computes the L2 distance between the robot's current
    `concat(eef_left_pos, eef_right_pos)` (6-dim) and the demo's
    matching-frame value, and emits the delta from the previous step:

        reward_t = (prev_dist - current_dist) * dist_coeff

    Pointer policy: increments by 1 from 0 each step. When the env
    episode is shorter than the demo, the pointer never reaches the
    end (we follow only the start of the demo, which is the approach
    phase). When longer, the pointer clips to the last demo frame
    (terminal pose).

    Args:
        demo_file (str): filename (relative to ``demo_data/``) OR an
            absolute path to a (T, 6) float32 numpy file containing the
            per-frame EEF positions of an expert trajectory.
        dist_coeff (float): scaling factor for the delta-distance
            term. Use ~1.0 with the default ``r_potential=1.0`` so the
            shaping shares a magnitude band with the existing
            ``PotentialReward`` of satisfied predicates.
    """

    def __init__(self, demo_file: str, dist_coeff: float):
        super().__init__()
        self._demo_file = demo_file
        self._dist_coeff = float(dist_coeff)
        self._demo_eef = self._load_demo(demo_file)
        self._demo_len = self._demo_eef.shape[0]
        self._step_idx = 0
        self._prev_dist = None

    @staticmethod
    def _load_demo(demo_file: str) -> th.Tensor:
        # Allow either an absolute path or a filename relative to the
        # bundled demo_data directory.
        p = Path(os.path.expanduser(demo_file))
        if not p.is_absolute():
            p = _DEMO_DATA_DIR / demo_file
        if not p.exists():
            raise FileNotFoundError(
                f"Demo EEF file not found: {p}. Place a (T, 6) float32 .npy "
                f"under {_DEMO_DATA_DIR} or pass an absolute path."
            )
        arr = np.load(p)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(
                f"Expected demo file shape (T, 6), got {arr.shape} from {p}"
            )
        return th.as_tensor(arr.astype(np.float32))

    def _current_eef(self, env) -> th.Tensor | None:
        """Concatenate left+right EEF positions of the first robot.

        Returns None if neither EEF can be read (defensive — the reward
        becomes a no-op rather than crashing).
        """
        robot = env.robots[0]
        arm_names = list(robot.arm_names) if hasattr(robot, "arm_names") else [robot.default_arm]
        eef_pieces = []
        for arm in arm_names[:2]:
            try:
                eef_pieces.append(robot.get_eef_position(arm))
            except Exception:
                pass
        if len(eef_pieces) == 0:
            return None
        # Single-arm robot: pad with the same value so both halves cancel.
        while len(eef_pieces) < 2:
            eef_pieces.append(eef_pieces[0])
        cur = th.cat([th.as_tensor(p, dtype=th.float32).flatten() for p in eef_pieces])
        return cur if cur.numel() == 6 else None

    def _step(self, task, env, action):
        cur_eef = self._current_eef(env)
        if cur_eef is None:
            return 0.0, {"demo_dist": 0.0, "demo_dist_reward": 0.0}

        idx = min(self._step_idx, self._demo_len - 1)
        dist = th.linalg.vector_norm(cur_eef - self._demo_eef[idx]).item()

        if self._prev_dist is None:
            reward = 0.0
        else:
            reward = (self._prev_dist - dist) * self._dist_coeff

        self._prev_dist = dist
        self._step_idx += 1
        return reward, {
            "demo_dist": dist,
            "demo_dist_reward": reward,
            "demo_step_idx": idx,
        }

    def reset(self, task, env):
        super().reset(task, env)
        self._step_idx = 0
        self._prev_dist = None
