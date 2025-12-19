"""Test VLA Diffusion Policy on Panda-Gym (Updated State Representation)"""

import os
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import gymnasium as gym
import panda_gym
import cv2  # Needed to resize images to match the model

from models.vla_diffusion_policy import VLADiffusionPolicy

# --- Simple Tokenizer ---
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        
    def encode(self, text):
        return [self.vocab.get(w, 1) for w in text.lower().split()]

def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Panda-Gym")
    
    # Note: Change the checkpoint filename if you saved it with a different name
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_panda_reach.pt",
                        help="Path to trained VLA diffusion checkpoint")
    parser.add_argument("--env-name", type=str, default="PandaReach-v3")
    parser.add_argument("--seed", type=int, default=100) # Change seed different from training for objective testing
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--instruction", type=str, default="move the gripper to the red target")
    parser.add_argument("--device", type=str, default="cuda", help="'cpu' or 'cuda'")
    
    parser.add_argument("--save-video", action="store_true", default=True, 
                        help="Save rollout videos")
    parser.add_argument("--video-dir", type=str, default="videos_test")

    return parser.parse_args()

def extract_state(obs_dict):
    """
    Helper function to extract state vector from observation dictionary.
    Structure MUST MATCH 100% with data collection.
    """
    return np.concatenate([
        obs_dict['observation'], 
        obs_dict['achieved_goal'], 
        obs_dict['desired_goal']  # <--- IMPORTANT: Added this
    ])

def load_model_and_tokenizer(checkpoint_path, device):
    print(f"[test] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_T = ckpt["diffusion_T"]
    
    resize_to = ckpt.get("resize_to", 64) 

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)

    return model, tokenizer, resize_to

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    try:
        model, tokenizer, resize_to = load_model_and_tokenizer(args.checkpoint, device)
        print(f"[test] Model loaded. Expecting image size: {resize_to}x{resize_to}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Encode Instruction
    instr_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # 3. Setup Environment
    env = gym.make(args.env_name, render_mode="rgb_array")
    
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    print(f"[test] Starting evaluation on {args.env_name}...")

    # 4. Evaluation Loop
    for ep in range(args.episodes):
        obs_dict, info = env.reset(seed=args.seed + ep)
        
        # --- Preprocess State ---
        # Use extract_state function to ensure desired_goal is included
        state = extract_state(obs_dict)
        
        img_raw = env.render()
        frames = [img_raw.copy()]
        
        step = 0
        ep_reward = 0.0
        done = False

        while not done and step < args.max_steps:
            # --- Preprocess Image ---
            img_resized = cv2.resize(img_raw, (resize_to, resize_to))
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            state_t = torch.from_numpy(state).float().unsqueeze(0)

            img_t = img_t.to(device)
            state_t = state_t.to(device)

            # --- Inference ---
            with torch.no_grad():
                action_t = model.act(img_t, text_ids, state_t)
            
            action_np = action_t.squeeze(0).cpu().numpy()

            # --- Step Environment ---
            obs_dict, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward
            step += 1

            # Update State and Image for next loop
            state = extract_state(obs_dict) # Call extract_state function again
            
            img_raw = env.render()
            frames.append(img_raw.copy())

        success = info.get('is_success', False)
        status_str = "SUCCESS" if success else "FAIL"
        print(f"[test] Episode {ep+1}/{args.episodes}: Steps={step}, Reward={ep_reward:.2f}, Status={status_str}")

        # --- Save Video ---
        if args.save_video:
            video_name = f"{args.env_name}_ep{ep+1:03d}_{status_str}.mp4"
            video_path = os.path.join(args.video_dir, video_name)
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)
            # print(f"Saved video to {video_path}")

    env.close()
    print("[test] Done.")

if __name__ == "__main__":
    main()