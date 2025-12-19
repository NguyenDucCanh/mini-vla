"""
Collect demonstration data from Panda-Gym environments using a calculated Expert.
Includes Video Recording functionality for the first 5 episodes.
"""

import os
import argparse
import time
import numpy as np
import gymnasium as gym
import panda_gym
import imageio  # Library for saving videos

# --- Simple Tokenizer ---
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.counter = 2
        
    def build_from_texts(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = self.counter
                    self.counter += 1
                    
    def encode(self, text):
        return [self.vocab.get(w, 1) for w in text.lower().split()]

# --- Arguments ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="PandaReach-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20) # Total number of datasets to collect
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-path", type=str, default="data/panda_reach_bc.npz")
    parser.add_argument("--instruction", type=str, default="move the gripper to the target",
                        help="Language instruction")
    
    # Parameters for video
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Enable video saving feature")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory for demo videos")
    
    return parser.parse_args()

# --- Expert Policy ---
def get_expert_action(obs):
    """
    Simple P-Controller for Reaching task
    """
    current_pos = obs['achieved_goal']
    target_pos = obs['desired_goal']
    error = target_pos - current_pos
    action = error * 5.0 
    action = np.clip(action, -1.0, 1.0)
    return action

# --- Main Loop ---
def main():
    args = parse_args()
    
    # Create output data directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Create video directory if needed
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # Initialize environment
    env = gym.make(args.env_name, render_mode="rgb_array")

    images = []
    states = []
    actions = []
    texts = []

    instruction = args.instruction
    print(f"Start collecting data for {args.env_name}...")
    print(f"Videos will be saved to: {args.video_dir}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        
        # Temporary list to contain frames of THIS SPECIFIC episode (for video)
        frames = [] 
        
        while not done and steps < args.max_steps:
            # 1. Get action
            action = get_expert_action(obs)

            # 2. Render & Collect Data
            img = env.render() # (720, 960, 3)
            
            # Save frame to buffer for video
            frames.append(img.copy()) 

            # Process state for dataset
            # --- EDIT THIS SECTION IN collect_panda_video.py ---
        # obs['desired_goal'] contains coordinates (x, y, z) of the red target point
        robot_state = np.concatenate([
            obs['observation'], 
            obs['achieved_goal'], 
            obs['desired_goal']  # <--- ADD THIS
        ])

            # Save to global dataset
            images.append(img.copy())
            states.append(robot_state.copy())
            actions.append(action.copy())
            texts.append(instruction)

            # 3. Step Env
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        print(f"Episode {ep+1}/{args.episodes} finished. Success: {info.get('is_success', False)}")

        # --- VIDEO SAVE LOGIC ---
        # Only save if save_video flag is on AND only save first 5 videos (ep < 5)
        if args.save_video and ep < 5:
            video_path = os.path.join(args.video_dir, f"{args.env_name}_ep{ep+1:03d}.mp4")
            # Use imageio to write mp4 file
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {video_path}")
        # -----------------------

    env.close()

    # --- Post-processing & Save Dataset ---
    print("Stacking arrays...")
    images_np = np.stack(images, axis=0)
    states_np = np.stack(states, axis=0)
    actions_np = np.stack(actions, axis=0)

    # Tokenize text
    tokenizer = SimpleTokenizer()
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    np.savez_compressed(
        args.output_path,
        images=images_np,
        states=states_np,
        actions=actions_np,
        text_ids=text_ids,
        vocab=tokenizer.vocab
    )

    print("-" * 30)
    print(f"Dataset saved to {args.output_path}")
    print(f"Total frames: {images_np.shape[0]}")

if __name__ == "__main__":
    main()