#!/usr/bin/env python3
"""
Run trained policy on SO-100 robot for inference.
Supports ACT, Diffusion, and Pi0-FAST policies.
"""

import time
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.robots.factory import make_robot

# =============================================================================
# CONFIGURATION - Update these for your setup
# =============================================================================
POLICY_PATH = "~/Desktop/act_40k"  # Path to your trained model
ROBOT_PORT = "/dev/tty.usbmodem5A7A0185761"  # Check with: ls /dev/tty.*
WRIST_CAM_INDEX = 0
FRONT_CAM_INDEX = 1
FPS = 30
# =============================================================================


def main():
    # Expand path
    policy_path = POLICY_PATH.replace("~", "/Users/krishnamgupta")
    
    # Load trained policy
    print(f"Loading policy from: {policy_path}")
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    
    # Use MPS (Apple Silicon) if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    policy = policy.to(device)
    
    # Connect to robot
    print(f"Connecting to robot on port: {ROBOT_PORT}")
    robot = make_robot(
        "so100_follower",
        robot_kwargs={"port": ROBOT_PORT},
        cameras={
            "wrist": {"type": "opencv", "index": WRIST_CAM_INDEX, "width": 640, "height": 480, "fps": FPS},
            "front": {"type": "opencv", "index": FRONT_CAM_INDEX, "width": 640, "height": 480, "fps": FPS},
        }
    )
    robot.connect()
    print("Robot connected!")
    
    print("\n" + "="*50)
    print("RUNNING POLICY - Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    try:
        step = 0
        while True:
            start_time = time.time()
            
            # Get observation from robot
            obs = robot.capture_observation()
            
            # Add batch dimension and move to device
            for key in obs:
                if isinstance(obs[key], torch.Tensor):
                    obs[key] = obs[key].unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(obs)
            
            # Send action to robot
            robot.send_action(action.squeeze(0).cpu())
            
            # Maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, (1/FPS) - elapsed)
            time.sleep(sleep_time)
            
            step += 1
            if step % 30 == 0:  # Print every second
                print(f"Step {step}, loop time: {elapsed*1000:.1f}ms")
                
    except KeyboardInterrupt:
        print("\n\nStopping policy execution...")
    finally:
        robot.disconnect()
        print("Robot disconnected. Done!")


if __name__ == "__main__":
    main()
