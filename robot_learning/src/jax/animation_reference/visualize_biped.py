"""
Simple script to visualize the biped XML file using MuJoCo's interactive viewer.
"""

import mujoco
import mujoco.viewer
import numpy as np
import os

# Get the path to the biped XML file
current_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(current_dir, '../../../assets')

xml_path = os.path.join(assets_dir, 'biped/xmls/biped_RL.xml')

# simple solution, not good
xml_path = "/home/marrodri/Documents/code-repositories/robot_learning_sorina/robot_learning/src/assets/biped/xmls/biped_RL.xml"
os.environ['MUJOCO_GL'] = 'egl'

def visualize_biped():
    """Load and visualize the biped model in an interactive MuJoCo viewer."""
    print(f"Loading XML file from: {xml_path}")
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to the "home" keyframe if it exists
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("Reset to keyframe 'home'")
    
    print(f"\nModel loaded successfully!")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"\nJoint names:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {joint_name}")
    
    print(f"\nOpening interactive viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view (left click + drag)")
    print("  - Mouse: Pan view (right click + drag)")
    print("  - Mouse: Zoom (scroll wheel)")
    print("  - Space: Pause/resume simulation")
    print("  - Ctrl + P: Slow motion")
    print("  - Right arrow: Step forward")
    print("  - Esc: Exit viewer")
    
    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run the simulation loop
        while viewer.is_running():
            # Step the simulation forward
            mujoco.mj_step(model, data)
            # Sync the viewer with the current state
            viewer.sync()
            
            # Optional: Add a small delay to control simulation speed
            import time
            time.sleep(0.01)

if __name__ == "__main__":
    visualize_biped()







