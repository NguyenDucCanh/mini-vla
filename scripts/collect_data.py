import os
import numpy as np
import torch

from envs.robot_kitchen_env import RobotKitchenWrapper
from utils.tokenizer import SimpleTokenizer


def scripted_expert_policy(state, target_qpos, action_dim):
    """
    Very simple proportional controller in joint-space.

    state:        vector; first k dims = joint positions
    target_qpos:  (k,) target joint angles (e.g., k=7)
    action_dim:   full env action_dim (e.g., 9 for FrankaKitchen)

    Returns: (action_dim,) action vector.
    """
    k = target_qpos.shape[0]
    q = state[:k]
    error = target_qpos - q
    Kp = 1.0
    u = Kp * error  # (k,)

    # Pad or trim to match env.action_dim
    if action_dim > k:
        pad = np.zeros(action_dim - k, dtype=np.float32)
        action = np.concatenate([u.astype(np.float32), pad], axis=-1)
    else:
        action = u.astype(np.float32)[:action_dim]

    return action

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="FrankaKitchen-v1")
    parser.add_argument("--output-path", type=str,
                        default="data/kitchen_imitation_dataset.npz")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    env = RobotKitchenWrapper(env_id=args.env_id)
    action_dim = env.action_dim
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tasks = [
        {"name": "open_microwave", "instruction": "open the microwave",
         "target_qpos": np.array([0.0, -0.5, 0.1, -1.2, 0.0, 0.8, 0.0], dtype=np.float32)},
        {"name": "turn_knob", "instruction": "turn the knob",
         "target_qpos": np.array([0.3, -0.2, 0.0, -1.0, 0.1, 0.5, 0.0], dtype=np.float32)},
    ]

    tokenizer = SimpleTokenizer(vocab=None)  # build from tasks or corpus
    all_images = []
    all_states = []
    all_actions = []
    all_text_ids = []

    num_episodes_per_task = 20
    max_steps = 100

    for task in tasks:
        instr = task["instruction"]
        target_qpos = task["target_qpos"]
        text_ids = tokenizer.encode(instr)  # e.g. List[int]

        for ep in range(num_episodes_per_task):
            img, state, _ = env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                action = scripted_expert_policy(state, target_qpos, action_dim)
                img_next, state_next, reward, done, info = env.step(action)

                all_images.append(img)        # current image
                all_states.append(state)
                all_actions.append(action)
                all_text_ids.append(text_ids)

                img = img_next
                state = state_next
                step += 1

    env.close()

    # Pad text sequences to fixed length
    max_len = max(len(t) for t in all_text_ids)
    text_ids_padded = np.zeros((len(all_text_ids), max_len), dtype=np.int64)
    for i, seq in enumerate(all_text_ids):
        text_ids_padded[i, :len(seq)] = np.array(seq, dtype=np.int64)

    images = np.stack(all_images, axis=0)          # (N, H, W, 3)
    states = np.stack(all_states, axis=0).astype(np.float32)
    actions = np.stack(all_actions, axis=0).astype(np.float32)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids_padded,
        vocab=tokenizer.vocab
    )
    print("Saved dataset with", images.shape[0], "samples to", args.output_path)


if __name__ == "__main__":
    main()
