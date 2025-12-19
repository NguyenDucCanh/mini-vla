"""
Test VLA Model with MAGIC GRASP enabled.
Purpose: Check if the model has learned to move and command to close the hand at the right time without being affected by physical errors (slipping).
"""

import os
import argparse
import numpy as np
import torch
import imageio
import gymnasium as gym
import panda_gym
import pybullet as p
import cv2 

# Import model class
from models.vla_diffusion_policy import VLADiffusionPolicy

# --- 1. CAMERA CONFIGURATION (MOST IMPORTANT) ---
# You must choose the Camera EXACTLY the same as when you Collect Data.
# If wrong camera, model will be "blind" and move randomly.

def get_camera_image(env, width=224, height=224):
    sim = env.unwrapped.task.sim
    physics_client = sim.physics_client 

    # =========================================================
    # OPTION 1: CORNER VIEW (Diagonal corner - Default of Magic Grasp code)
    # 
    camera_eye_pos = [-0.2, -0.6, 0.6] 
    
    # OPTION 2: HUMANOID VIEW (Human view - If you trained with old file)
    # Uncomment the line below if you trained with Humanoid data!
    # camera_eye_pos = [-0.6, 0.0, 0.8] 
    # =========================================================

    camera_target_pos = [0.0, 0.0, 0.0] 
    camera_up_vector = [0, 0, 1] 

    view_matrix = physics_client.computeViewMatrix(
        cameraEyePosition=camera_eye_pos,
        cameraTargetPosition=camera_target_pos,
        cameraUpVector=camera_up_vector
    )
    proj_matrix = physics_client.computeProjectionMatrixFOV(
        fov=50, aspect=width/height, nearVal=0.1, farVal=100.0
    )
    width, height, rgb, depth, seg = physics_client.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = np.array(rgb, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))
    rgb_array = rgb_array[:, :, :3] 
    return rgb_array

# --- 2. MAGIC GRASP MANAGER (Grip support) ---
class MagicGraspManager:
    def __init__(self, env):
        self.env = env
        self.sim = env.unwrapped.task.sim
        self.constraint_id = None
        self.robot_body_id = self.sim._bodies_idx['panda'] 
        self.object_body_id = self.sim._bodies_idx['object']
        self.ee_link_id = 11 # Gripper tip link

    def attach_object(self):
        """Only activate if robot hand is ALREADY CLOSE to object (< 5cm)"""
        if self.constraint_id is not None: return 

        # Get real position from PyBullet to check distance
        robot_pos = self.sim.get_link_position("panda", 11)
        obj_pos = self.sim.get_base_position("object")
        dist = np.linalg.norm(np.array(robot_pos) - np.array(obj_pos))
        
        # If model controls hand close to object then allow Magic Grasp
        if dist < 0.05: 
            self.constraint_id = self.sim.physics_client.createConstraint(
                parentBodyUniqueId=self.robot_body_id,
                parentLinkIndex=self.ee_link_id,
                childBodyUniqueId=self.object_body_id,
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            self.sim.physics_client.changeConstraint(self.constraint_id, maxForce=2000)

    def detach_object(self):
        if self.constraint_id is not None:
            self.sim.physics_client.removeConstraint(self.constraint_id)
            self.constraint_id = None

# --- 3. UTILITY FUNCTIONS ---
class SimpleTokenizer:
    def __init__(self, vocab): self.vocab = vocab
    def encode(self, text): return [self.vocab.get(w, 1) for w in text.lower().split()]

def extract_state(obs):
    # Concatenate state exactly as during training
    return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])

def load_model(checkpoint_path, device):
    print(f"[Test] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found!")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = VLADiffusionPolicy(
        vocab_size=max(ckpt["vocab"].values()) + 1,
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        d_model=ckpt["d_model"],
        diffusion_T=ckpt["diffusion_T"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Get the image size that the model requires
    resize_to = ckpt.get("resize_to", 64) 
    return model, ckpt["vocab"], resize_to

# --- 4. MAIN LOOP ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--episodes", type=int, default=5) # Test 5 videos
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--video-dir", type=str, default="videos_test_magic")
    parser.add_argument("--instruction", type=str, default="pick up the block and move it to the goal")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.save_video: 
        os.makedirs(args.video_dir, exist_ok=True)

    # Load Model
    model, vocab, resize_to = load_model(args.checkpoint, device)
    print(f"[Test] Model expects image size: {resize_to}x{resize_to}")
    
    tokenizer = SimpleTokenizer(vocab)
    text_ids = torch.tensor(tokenizer.encode(args.instruction)).long().unsqueeze(0).to(device)

    # Init Env
    env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=args.max_steps)
    magic_manager = MagicGraspManager(env)

    print(f"[Test] Starting {args.episodes} episodes with Magic Grasp...")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=2000 + ep)
        magic_manager.detach_object() # Reset magic
        
        frames = []
        done = False
        steps = 0
        
        # Render first frame
        img_raw = get_camera_image(env)
        if args.save_video: frames.append(img_raw.copy())

        while not done and steps < args.max_steps:
            # --- 1. PREPARE INPUT DATA ---
            # Resize image to model required size (e.g. 64x64 or 224x224)
            img_resized = cv2.resize(img_raw, (resize_to, resize_to))
            
            # Convert to Tensor (Normalize 0-1)
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            state_t = torch.from_numpy(extract_state(obs)).float().unsqueeze(0).to(device)
            
            # --- 2. MODEL MAKES DECISION ---
            with torch.no_grad():
                # Model returns action (1, 4)
                action_tensor = model.act(img_t, text_ids, state_t)
            
            action = action_tensor.squeeze(0).cpu().numpy()

            # --- 3. HANDLE MAGIC GRASP ---
            # action[3] is gripper control.
            # If model commands < -0.5 (Close hand) -> Activate Magic
            if action[3] < -0.5:
                magic_manager.attach_object()
            else:
                magic_manager.detach_object() # If model commands open hand (> -0.5) -> Drop object

            # --- 4. STEP ENV ---
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Render new image
            img_raw = get_camera_image(env)
            if args.save_video: frames.append(img_raw.copy())
            
            # Check success
            if info.get('is_success'): 
                # If successful, run additional 10 steps for better video then stop
                if steps > args.max_steps - 10: 
                    done = True
            
            if terminated or truncated: 
                done = True

        status = "SUCCESS" if info.get('is_success') else "FAIL"
        print(f"Episode {ep+1}: Steps={steps}, Status={status}")
        
        # Save Video
        if args.save_video:
            video_path = os.path.join(args.video_dir, f"test_ep{ep+1:03d}_{status}.mp4")
            try:
                imageio.mimsave(video_path, frames, fps=20)
                print(f"  --> Saved video: {video_path}")
            except Exception as e:
                print(f"  --> Error saving video: {e}")

    env.close()

if __name__ == "__main__":
    main()