# envs/libero_env.py

import os
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Any

import numpy as np

from LIBERO.libero.libero import benchmark, get_libero_path
from LIBERO.libero.libero.envs import OffScreenRenderEnv


@dataclass
class LiberoEnvConfig:
    """
    Configuration for a single LIBERO task.

    benchmark_name: one of
        - "libero_spatial"
        - "libero_object"
        - "libero_goal"
        - "libero_90"
        - "libero_10"

    task_id: zero-based index into the benchmark's tasks
    """
    benchmark_name: str = "libero_spatial"
    task_id: int = 0
    camera_width: int = 128
    camera_height: int = 128

    # Which keys from the LIBERO observation dict to pack into the state vector.
    # By default: joint positions + gripper qpos.
    state_keys: Tuple[str, ...] = (
        "robot0_joint_pos",
        "robot0_gripper_qpos",
    )

    # LIBERO cameras are often upside-down vs training setups;
    # set to False if you DON'T want rotation.
    rotate_image_180: bool = True


class LiberoEnvWrapper:
    """
    Thin, VLA-friendly wrapper around LIBERO's OffScreenRenderEnv.

    Exposes:
        - reset() -> (img, state)
        - step(action) -> (img, state, reward, done, info)
        - close()

    img:   H x W x 3 uint8 (agentview RGB)
    state: 1D float32 vector built from cfg.state_keys
    """

    def __init__(self, cfg: LiberoEnvConfig):
        self.cfg = cfg

        # 1) Get the task suite
        bench_dict = benchmark.get_benchmark_dict()
        if cfg.benchmark_name not in bench_dict:
            raise ValueError(
                f"Unknown benchmark '{cfg.benchmark_name}'. "
                f"Available: {list(bench_dict.keys())}"
            )

        self.task_suite = bench_dict[cfg.benchmark_name]()
        self.task = self.task_suite.get_task(cfg.task_id)

        # 2) Resolve BDDL path and initial states
        bddl_root = get_libero_path("bddl_files")
        self.bddl_file = os.path.join(
            bddl_root,
            self.task.problem_folder,
            self.task.bddl_file,
        )

        self.init_states = self.task_suite.get_task_init_states(cfg.task_id)

        # 3) Build underlying env
        env_args = dict(
            bddl_file_name=self.bddl_file,
            camera_heights=self.cfg.camera_height,
            camera_widths=self.cfg.camera_width,
        )
        self.env = OffScreenRenderEnv(**env_args)

        # LIBERO uses 7-D continuous action by default (OSC pose + gripper)
        self.action_dim = 7

        # 4) Probe a reset to infer shapes
        img, state = self.reset()
        self.obs_shape = img.shape
        self.state_dim = state.shape[0]

    # ----------------- public API -----------------

    def reset(self, init_state_idx: int = 0):
        """
        Reset env and set one of LIBERO's canonical initial states.
        Returns (rgb_image, state_vector).
        """
        self.env.reset()

        init_state = self.init_states[init_state_idx]
        obs = self.env.set_init_state(init_state)

        img = self._extract_image(obs)
        state = self._extract_state(obs)

        return img, state

    def step(self, action: np.ndarray):
        """
        Step with a 7-D continuous action.

        Returns (rgb_image, state_vector, reward, done, info)
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        obs, reward, done, info = self.env.step(action)

        img = self._extract_image(obs)
        state = self._extract_state(obs)

        return img, state, float(reward), bool(done), info

    def close(self):
        self.env.close()

    # ----------------- helpers -----------------

    def _extract_image(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Take the main agentview RGB image and (optionally) rotate 180°.
        """
        if "agentview_image" not in obs:
            raise KeyError(
                f"'agentview_image' not found in obs. "
                f"Available keys: {list(obs.keys())}"
            )

        img = obs["agentview_image"]
        img = np.array(img, copy=False)

        if self.cfg.rotate_image_180:
            # LIBERO camera is inverted in many setups
            img = img[::-1, ::-1]

        return img

    def _extract_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Build a low-dim state vector by concatenating selected obs fields.
        Default is [robot0_joint_pos, robot0_gripper_qpos].
        """
        parts = []
        for key in self.cfg.state_keys:
            if key not in obs:
                raise KeyError(
                    f"State key '{key}' not found in obs. "
                    f"Available keys: {list(obs.keys())}"
                )
            v = np.asarray(obs[key], dtype=np.float32).ravel()
            parts.append(v)

        state = np.concatenate(parts, axis=0)
        return state
