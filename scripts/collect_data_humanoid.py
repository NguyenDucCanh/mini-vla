"""
Collect Data PandaPickAndPlace-v3
Technique: MAGIC GRASP (PyBullet Constraint) -> 100% Success Rate
Output: Dataset .npz & 5 Demo Videos
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import panda_gym
import pybullet as p
import imageio

# --- 1. Tokenizer (Process text instruction) ---
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

# --- 2. Camera Setup (Diagonal view from top down) ---
def get_corner_camera_image(env, width=224, height=224):
    sim = env.unwrapped.task.sim
    physics_client = sim.physics_client 

    # Camera position: Back (-x), Right (-y), Up (+z)
    camera_eye_pos = [-0.2, -0.6, 0.6] 
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
    
    # Convert to numpy array (H, W, 3)
    rgb_array = np.array(rgb, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))
    rgb_array = rgb_array[:, :, :3] # Remove Alpha channel
    return rgb_array

# --- 3. Magic Grasp Manager (Create virtual magnet) ---
class MagicGraspManager:
    def __init__(self, env):
        self.env = env
        self.sim = env.unwrapped.task.sim
        self.constraint_id = None
        
        # IDs of Robot and Object in PyBullet
        self.robot_body_id = self.sim._bodies_idx['panda'] 
        self.object_body_id = self.sim._bodies_idx['object']
        self.ee_link_id = 11 # Panda gripper tip link

    def attach_object(self):
        """Lock object tightly to robot hand"""
        if self.constraint_id is not None: return 

        # Create point-to-point constraint
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
        """Unlock (Drop object)"""
        if self.constraint_id is not None:
            self.sim.physics_client.removeConstraint(self.constraint_id)
            self.constraint_id = None

# --- 4. Waypoint Expert Policy ---
class WaypointExpert:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.stage = 0
        self.timer = 0
        
    def get_action(self, obs):
        ee_pos = obs['observation'][0:3]
        object_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']
        
        action = np.zeros(4)
        kp = 10.0 # Movement speed
        
        # Stage 0: Fly to above object
        if self.stage == 0:
            target = object_pos.copy()
            target[2] = 0.1
            error = target - ee_pos
            action[:3] = error * kp
            action[3] = 1.0 # Open hand
            if np.linalg.norm(error) < 0.02: self.stage = 1

        # Stage 1: Lower to grasp position
        elif self.stage == 1:
            target = object_pos.copy()
            target[2] = 0.01 # Lower close to object
            error = target - ee_pos
            action[:3] = error * kp
            action[3] = 1.0 
            if np.linalg.norm(error) < 0.01: self.stage = 2

        # Stage 2: Close hand (Magic Grasp will activate in Main)
        elif self.stage == 2:
            action[:3] = [0,0,0]
            action[3] = -1.0 # Close hand
            self.timer += 1
            if self.timer > 5: self.stage = 3 # Only need to wait 5 steps because Magic is available

        # Stage 3: Lift up
        elif self.stage == 3:
            target = ee_pos.copy()
            target[2] = 0.2
            error = target - ee_pos
            action[:3] = error * kp
            action[3] = -1.0 # Keep closed
            if np.linalg.norm(error) < 0.02: self.stage = 4
            
        # Stage 4: Fly to goal
        elif self.stage == 4:
            target = goal_pos.copy()
            error = target - ee_pos
            action[:3] = error * kp
            action[3] = -1.0
            if np.linalg.norm(error) < 0.02: self.stage = 5
            
        # Stage 5: Drop object
        elif self.stage == 5:
            action[:3] = [0,0,0]
            action[3] = 1.0 # Open hand

        return np.clip(action, -1.0, 1.0)

# --- 5. Main Script ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100) # Number of samples to collect
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output-path", type=str, default="data/magic_grasp.npz")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--video-dir", type=str, default="videos_magic")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.save_video: os.makedirs(args.video_dir, exist_ok=True)

    # Init Env
    env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=args.max_steps)
    
    expert = WaypointExpert(env)
    magic_manager = MagicGraspManager(env)

    images, states, actions, texts = [], [], [], []
    instruction = "pick up the block and move it to the goal"
    success_count = 0
    
    # VIDEO SAVE CONFIG: Only save first 5 episodes
    MAX_VIDEOS_TO_SAVE = 5

    print(f"Collecting Data ({args.episodes} episodes)...")
    print(f"Videos will be saved to: {args.video_dir}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        expert.reset()
        magic_manager.detach_object() # Reset magic state
        
        frames = [] # Buffer for images of this specific episode
        steps = 0
        done = False
        
        while not done and steps < args.max_steps:
            action = expert.get_action(obs)
            
            # --- MAGIC GRASP LOGIC ---
            # If in Lift (3) or Move (4) stage -> Lock object
            if expert.stage in [3, 4]:
                magic_manager.attach_object()
            
            # If drop (5) -> Unlock
            if expert.stage == 5:
                magic_manager.detach_object()
            # -------------------------

            # Render image
            img = get_corner_camera_image(env)
            
            # Save Data
            robot_state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            images.append(img.copy())
            states.append(robot_state.copy())
            actions.append(action.copy())
            texts.append(instruction)
            
            # IMPORTANT LOGIC: Only add frame to buffer if video needs to be saved
            if args.save_video and ep < MAX_VIDEOS_TO_SAVE:
                frames.append(img.copy())

            # Step Env
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Check Success & Early Stop
            if info.get('is_success', False):
                if steps > args.max_steps - 10: done = True
            
            if terminated or truncated: done = True

        status = "SUCCESS" if info.get('is_success') else "FAIL"
        if status == "SUCCESS": success_count += 1
        
        print(f"Ep {ep+1}/{args.episodes}: Steps={steps}, Status={status}")

        # SAVE VIDEO TO FILE
        if args.save_video and ep < MAX_VIDEOS_TO_SAVE:
            video_name = f"magic_ep{ep+1:03d}_{status}.mp4"
            video_path = os.path.join(args.video_dir, video_name)
            try:
                # Use imageio.mimsave to write MP4 file
                imageio.mimsave(video_path, frames, fps=20)
                print(f"  --> Saved video: {video_path}")
            except Exception as e:
                print(f"  --> Error saving video: {e}")

    env.close()
    
    # Save Final Dataset
    print("-" * 30)
    print(f"Collection Finished. Total Success: {success_count}/{args.episodes}")
    print("Saving Dataset to .npz ...")
    
    images_np = np.stack(images)
    states_np = np.stack(states)
    actions_np = np.stack(actions)
    
    # Tokenize Text
    tokenizer = SimpleTokenizer()
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(t) for t in text_ids_list)
    text_ids_np = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(text_ids_list):
        text_ids_np[i, :len(t)] = t

    np.savez_compressed(
        args.output_path,
        images=images_np, 
        states=states_np, 
        actions=actions_np,
        text_ids=text_ids_np, 
        vocab=tokenizer.vocab, 
        resize_to=224 # Save this parameter so that train/test knows the image size
    )
    print("Done.")

if __name__ == "__main__":
    main()