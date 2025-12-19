"""
Collect Data PandaPickAndPlace-v3
Style: Meta-World like Waypoint Policy (More Robust)
View: Corner View
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import panda_gym
import pybullet as p
import imageio

# --- 1. Tokenizer ---
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

# --- 2. Camera: Corner View ---
def get_corner_camera_image(env, width=224, height=224):
    sim = env.unwrapped.task.sim
    physics_client = sim.physics_client 

    # Comprehensive view from left/right corner
    camera_eye_pos = [-0.2, -0.8, 0.8] 
    camera_target_pos = [0.0, 0.0, 0.0] 
    camera_up_vector = [0, 0, 1] 

    view_matrix = physics_client.computeViewMatrix(
        cameraEyePosition=camera_eye_pos,
        cameraTargetPosition=camera_target_pos,
        cameraUpVector=camera_up_vector
    )
    proj_matrix = physics_client.computeProjectionMatrixFOV(
        fov=60, aspect=width/height, nearVal=0.1, farVal=100.0
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

# --- 3. Waypoint Expert Policy (Meta-World style) ---
class WaypointExpert:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.current_waypoint_idx = 0
        self.gripper_wait = 0
        
    def get_action(self, obs):
        # Get information
        ee_pos = obs['observation'][0:3]
        object_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']
        
        # Define Waypoints (Movement milestones)
        # P0: Above object (Pre-grasp)
        p0 = object_pos.copy()
        p0[2] += 0.1
        
        # P1: Grasp position
        p1 = object_pos.copy()
        # Important: Lower slightly below object center for firm grip
        p1[2] -= 0.002 
        
        # P2: Lift position
        p2 = object_pos.copy()
        p2[2] += 0.2
        
        # P3: Goal position
        p3 = goal_pos.copy()

        # Waypoint transition logic
        error = 0
        target = ee_pos
        gripper = 1.0 # 1.0 = Open, -1.0 = Close
        
        # --- WAYPOINT 0: Move to above object ---
        if self.current_waypoint_idx == 0:
            target = p0
            gripper = 1.0
            if np.linalg.norm(target - ee_pos) < 0.05:
                self.current_waypoint_idx = 1
                
        # --- WAYPOINT 1: Lower to grasp ---
        elif self.current_waypoint_idx == 1:
            target = p1
            gripper = 1.0
            if np.linalg.norm(target - ee_pos) < 0.01: # Need precision
                self.current_waypoint_idx = 2
                self.gripper_wait = 0 # Start waiting timer
                
        # --- WAYPOINT 2: Close gripper and Wait (Stationary) ---
        elif self.current_waypoint_idx == 2:
            target = p1 # Keep position
            gripper = -1.0 # CLOSE
            self.gripper_wait += 1
            
            # Wait 25 steps for physics stabilization (Meta-world doesn't need this but Bullet does)
            if self.gripper_wait > 25: 
                self.current_waypoint_idx = 3
                
        # --- WAYPOINT 3: Lift up ---
        elif self.current_waypoint_idx == 3:
            # Lift vertically from current position
            target = np.array([ee_pos[0], ee_pos[1], 0.3])
            gripper = -1.0 # Keep closed
            if np.linalg.norm(target - ee_pos) < 0.05:
                self.current_waypoint_idx = 4
                
        # --- WAYPOINT 4: Move to Goal ---
        elif self.current_waypoint_idx == 4:
            target = p3
            gripper = -1.0
            # No need to change state anymore, stay here
        
        # Calculate Action (P-Controller)
        # If in grasp state (2), small Kp to avoid bouncing
        kp = 10.0
        if self.current_waypoint_idx == 2: 
            kp = 2.0 
            
        pos_error = target - ee_pos
        action = np.zeros(4)
        action[:3] = pos_error * kp
        action[3] = gripper
        
        return np.clip(action, -1.0, 1.0)

# --- 4. Main Script ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="PandaPickAndPlace-v3")
    parser.add_argument("--episodes", type=int, default=400) 
    parser.add_argument("--max-steps", type=int, default=200) # INCREASE TO 200
    parser.add_argument("--output-path", type=str, default="data/humanoid_grasp.npz")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--video-dir", type=str, default="videos_corner")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.save_video: 
        os.makedirs(args.video_dir, exist_ok=True)

    # Important: Override max_episode_steps to 200
    env = gym.make(args.env_name, render_mode="rgb_array", max_episode_steps=args.max_steps)
    
    expert = WaypointExpert(env)

    images = []
    states = []
    actions = []
    texts = []
    instruction = "pick up the block and move it to the goal"

    print(f"Collecting Data ({args.episodes} episodes) - Waypoint Policy...")
    print(f"Videos saved to: {args.video_dir}")
    
    success_count = 0
    MAX_VIDEO_SAVE = 5 

    for ep in range(args.episodes):
        obs, info = env.reset(seed=500 + ep)
        expert.reset()
        
        frames = [] 
        steps = 0
        done = False
        
        while not done and steps < args.max_steps:
            # Action
            action = expert.get_action(obs)
            
            # Render
            img = get_corner_camera_image(env, width=224, height=224)
            
            # State
            robot_state = np.concatenate([
                obs['observation'], 
                obs['achieved_goal'],
                obs['desired_goal']
            ])

            images.append(img.copy())
            states.append(robot_state.copy())
            actions.append(action.copy())
            texts.append(instruction)
            
            if args.save_video and ep < MAX_VIDEO_SAVE:
                frames.append(img.copy())

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            is_success = info.get('is_success', False)
            
            # If successful, stay there for 10 more steps then done
            if is_success:
                # Force robot to stay at Goal (waypoint 4)
                expert.current_waypoint_idx = 4 
                if steps > args.max_steps - 10: 
                    done = True
            
            if terminated or truncated:
                done = True

        final_success = info.get('is_success', False)
        if final_success: success_count += 1
        
        status = "SUCCESS" if final_success else "FAIL"
        print(f"Ep {ep+1}: Steps={steps}, Status={status}")

        if args.save_video and ep < MAX_VIDEO_SAVE:
            video_name = f"corner_ep{ep+1:03d}_{status}.mp4"
            video_path = os.path.join(args.video_dir, video_name)
            try:
                with imageio.get_writer(video_path, fps=20) as writer:
                    for f in frames: writer.append_data(f)
            except: pass

    env.close()

    print(f"\nFinished. Success Rate: {success_count}/{args.episodes}")

    # Save
    print("Saving NPZ...")
    images_np = np.stack(images, axis=0)
    states_np = np.stack(states, axis=0)
    actions_np = np.stack(actions, axis=0)
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max([len(t) for t in text_ids_list])
    padded_text = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(text_ids_list):
        padded_text[i, :len(t)] = np.array(t)

    np.savez_compressed(
        args.output_path,
        images=images_np,
        states=states_np,
        actions=actions_np,
        text_ids=padded_text,
        vocab=tokenizer.vocab,
        resize_to=224
    )
    print("Done.")

if __name__ == "__main__":
    main()