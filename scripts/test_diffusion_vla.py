import argparse
import numpy as np
import torch

from envs.robot_kitchen_env import RobotKitchenWrapper
from envs.libero_env import LiberoEnvWrapper, LiberoEnvConfig
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

import os
import imageio.v2 as imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/vla_diffusion.pt")
    parser.add_argument("--env-id", type=str, default="FrankaKitchen-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--instruction", type=str,
                        default="open the microwave")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-video",
        action="store_true", help="If set, save each episode as an MP4 video")
    parser.add_argument("--video-dir",type=str,
        default="videos",help="Directory to save episode videos")
    return parser.parse_args()

cfg = LiberoEnvConfig(
    benchmark_name="libero_spatial",
    task_id=0,
    camera_width=128,
    camera_height=128,
    # Adjust this to match how you built `state` in your LIBERO NPZ
    state_keys=("robot0_joint_pos", "robot0_gripper_qpos"),
)

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_T = ckpt["diffusion_T"]

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Tokenizer consistent with training
    tokenizer = SimpleTokenizer(vocab=vocab)
    text_ids = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, T_text)

    # env = RobotKitchenWrapper(env_id=args.env_id)
    env = LiberoEnvWrapper(cfg)
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
    for ep in range(args.episodes):
        img, state = env.reset()
        done = False
        step = 0
        ep_reward = 0.0

        frames = []
        frames.append(img)

        while not done and step < args.max_steps:
            # preprocess image / state
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            state_t = torch.from_numpy(state).float().unsqueeze(0)

            img_t = img_t.to(device)
            state_t = state_t.to(device)

            with torch.no_grad():
                action_t = model.act(img_t, text_ids, state_t)  # (1, action_dim)
            action_np = action_t.squeeze(0).cpu().numpy()

            img, state, reward, done, info = env.step(action_np)
            ep_reward += reward
            step += 1

            frames.append(img)

        print(f"Episode {ep+1}: total reward = {ep_reward:.3f}, steps = {step}")

        if args.save_video:
            video_path = os.path.join(args.video_dir, f"episode_{ep+1}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Saved video: {video_path}")

    env.close()


if __name__ == "__main__":
    main()
