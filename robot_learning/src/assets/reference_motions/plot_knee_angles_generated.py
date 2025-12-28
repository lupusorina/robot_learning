#!/usr/bin/env python3
"""
Script to plot knee angles from the humanoid_walk.pkl motion file.

The motion file contains frame data with structure:
  - frames: [num_frames, 6 + num_joints] array
            First 6 values: root_pos(3) + root_rot(3)
            Remaining values: joint DOF angles
  - fps: frames per second
  - loop_mode: 0 (CLAMP) or 1 (WRAP)

Usage:
    # Basic usage (single cycle):
    python plot_knee_angles.py
    
    # Extend timeframe by looping the motion multiple times:
    python plot_knee_angles.py --num_loops 5
    
    # Or specify exact number of steps/frames:
    python plot_knee_angles.py --num_steps 500
    
    # Or specify target duration in seconds:
    python plot_knee_angles.py --duration 5.0

The script will automatically detect valid knee indices if the provided ones
contain NaN values. To manually specify different indices, modify the
KNEE_JOINT_INDICES variable below or use --knee_indices.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import mujoco

# Path to the motion file
MOTION_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "humanoid_walk.pkl"
)

# Path to the humanoid XML file
HUMANOID_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "characters",
    "humanoid.xml"
)

# Knee joint indices in the joint DOF array (after root_pos and root_rot)
# Format: [left_knee_idx, right_knee_idx]
# These will be automatically determined from the XML file, but can be overridden
KNEE_JOINT_INDICES = None  # Will be set from XML file

# Default number of loops (set to 1 for single cycle, increase to see longer periods)
DEFAULT_NUM_LOOPS = 1

def load_motion_data(file_path):
    """Load motion data from pickle file."""
    with open(file_path, 'rb') as f:
        motion_data = pickle.load(f)
    return motion_data

def get_knee_indices_from_xml(xml_path):
    """
    Parse the humanoid XML file to get the correct indices for left_knee and right_knee joints.
    
    In MuJoCo motion files, the frame structure is: [root_pos(3), root_rot(3), joint_dofs...]
    The joint_dofs start at index 0 after the first 6 values (root_pos + root_rot).
    Joint DOFs are ordered by joint qposadr, but we need to account for the freejoint offset.
    
    Args:
        xml_path: Path to the humanoid.xml file
    
    Returns:
        Tuple of (left_knee_idx, right_knee_idx) in the joint DOF array (after root_pos and root_rot)
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # Find the freejoint (root) to get its qpos size (usually 7: 3 pos + 4 quat)
    freejoint_qpos_size = 0
    for i in range(model.njnt):
        joint_type = model.jnt_type[i]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            # Freejoint uses 7 DOFs (3 pos + 4 quat) in qpos
            freejoint_qpos_size = 7
            break
    
    # Get qpos addresses for knee joints
    try:
        left_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        right_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        
        if left_knee_joint_id == -1 or right_knee_joint_id == -1:
            raise ValueError(f"Could not find knee joints in XML")
        
        # Get qpos addresses (positions in the full qpos array)
        left_knee_qpos_adr = model.jnt_qposadr[left_knee_joint_id]
        right_knee_qpos_adr = model.jnt_qposadr[right_knee_joint_id]
        
        # In motion files: [root_pos(3), root_rot(3), ...joint_dofs]
        # The root uses 3 pos + 3 rot (Euler, not quat) = 6 values
        # So joint DOF index = qpos_adr - freejoint_qpos_size + 3 (since root_pos is 3, root_rot is 3)
        # Actually, motion files typically use: root_pos(3) + root_rot(3) + joint_qpos (excluding root)
        # So the offset is 6 (root_pos + root_rot), and we subtract the freejoint qpos size
        
        # More accurately: motion frame = [root_pos(3), root_rot(3), joint_qpos_without_root]
        # joint_qpos_without_root starts after the freejoint, so index = qpos_adr - freejoint_qpos_size
        left_knee_idx = left_knee_qpos_adr - freejoint_qpos_size
        right_knee_idx = right_knee_qpos_adr - freejoint_qpos_size
        
        # Get joint names for debugging
        all_joint_names = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                all_joint_names.append(name)
        
        print(f"Found knee joint indices from XML:")
        print(f"  Freejoint qpos size: {freejoint_qpos_size}")
        print(f"  Left knee qpos address: {left_knee_qpos_adr}, joint DOF index: {left_knee_idx}")
        print(f"  Right knee qpos address: {right_knee_qpos_adr}, joint DOF index: {right_knee_idx}")
        print(f"  All joints: {all_joint_names}")
        
        return left_knee_idx, right_knee_idx
        
    except Exception as e:
        # Fallback: try to find by iterating through joints
        print(f"Error using qposadr method: {e}")
        print("Attempting alternative method...")
        
        joint_names = []
        joint_dof_indices = {}
        current_dof_idx = 0
        
        for i in range(model.njnt):
            joint_type = model.jnt_type[i]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            
            if joint_name:
                joint_names.append(joint_name)
                
                # Skip freejoint (it's represented by root_pos and root_rot in motion file)
                if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                    joint_dof_indices[joint_name] = current_dof_idx
                    # Hinge joints have 1 DOF, ball joints have 3, etc.
                    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                        current_dof_idx += 1
                    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                        current_dof_idx += 3
                    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                        current_dof_idx += 1
                    # Add other joint types as needed
        
        left_knee_idx = joint_dof_indices.get('left_knee')
        right_knee_idx = joint_dof_indices.get('right_knee')
        
        if left_knee_idx is None or right_knee_idx is None:
            raise ValueError(f"Could not find knee joints. Available joints: {list(joint_dof_indices.keys())}")
        
        print(f"Found knee joint indices (alternative method):")
        print(f"  Left knee index: {left_knee_idx}")
        print(f"  Right knee index: {right_knee_idx}")
        
        return left_knee_idx, right_knee_idx

def find_knee_indices(joint_dof, left_knee_idx, right_knee_idx):
    """
    Find valid knee indices, trying alternatives if provided indices have NaN values.
    
    Args:
        joint_dof: Joint DOF array [num_frames, num_joints]
        left_knee_idx: Initial guess for left knee index
        right_knee_idx: Initial guess for right knee index
    
    Returns:
        Tuple of (left_knee_idx, right_knee_idx) that work
    """
    # Check if provided indices are valid
    left_valid = not np.any(np.isnan(joint_dof[:, left_knee_idx])) if left_knee_idx < joint_dof.shape[1] else False
    right_valid = not np.any(np.isnan(joint_dof[:, right_knee_idx])) if right_knee_idx < joint_dof.shape[1] else False
    
    if left_valid and right_valid:
        return left_knee_idx, right_knee_idx
    
    # Try to find alternative indices
    print(f"WARNING: Provided indices [{left_knee_idx}, {right_knee_idx}] have NaN values.")
    print("Attempting to find valid knee indices...")
    
    # Find all indices with valid data and reasonable variation (potential knee joints)
    valid_indices = []
    for i in range(joint_dof.shape[1]):
        if not np.any(np.isnan(joint_dof[:, i])):
            std_val = np.std(joint_dof[:, i])
            min_val, max_val = np.min(joint_dof[:, i]), np.max(joint_dof[:, i])
            # Look for joints with some variation (std > 0.01) and reasonable range
            if std_val > 0.01 and -3.0 <= min_val <= 3.0 and -3.0 <= max_val <= 3.0:
                valid_indices.append((i, std_val, min_val, max_val))
    
    if len(valid_indices) >= 2:
        # Sort by standard deviation (knee joints typically have good variation)
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        # Try the two indices with highest variation as knee candidates
        alt_left = valid_indices[0][0]
        alt_right = valid_indices[1][0] if len(valid_indices) > 1 else valid_indices[0][0]
        print(f"  Found alternative indices: [{alt_left}, {alt_right}]")
        print(f"    Index {alt_left}: std={valid_indices[0][1]:.3f}, range=[{valid_indices[0][2]:.3f}, {valid_indices[0][3]:.3f}]")
        if len(valid_indices) > 1:
            print(f"    Index {alt_right}: std={valid_indices[1][1]:.3f}, range=[{valid_indices[1][2]:.3f}, {valid_indices[1][3]:.3f}]")
        
        # Use alternative if original doesn't work
        if not left_valid:
            left_knee_idx = alt_left
        if not right_valid:
            right_knee_idx = alt_right if alt_right != left_knee_idx else valid_indices[2][0] if len(valid_indices) > 2 else alt_right
    else:
        print("  Could not find suitable alternative indices. Using provided indices (may contain NaN).")
    
    return left_knee_idx, right_knee_idx

def extract_knee_angles(motion_data, knee_indices, num_loops=1, num_steps=None, duration=None):
    """
    Extract knee angles from motion data, optionally looping to extend timeframe.
    
    Args:
        motion_data: Dictionary with 'frames', 'fps', 'loop_mode'
        knee_indices: List [left_knee_idx, right_knee_idx] in joint DOF array
        num_loops: Number of times to repeat the motion (default: 1)
        num_steps: Optional exact number of steps to generate (overrides num_loops)
        duration: Optional target duration in seconds (overrides num_loops and num_steps)
    
    Returns:
        Dictionary with time array and knee angles
    """
    frames = np.array(motion_data['frames'])  # [num_frames, 6 + num_joints]
    fps = motion_data['fps']
    
    # Extract joint DOFs (everything after root_pos(3) + root_rot(3))
    joint_dof = frames[:, 6:]  # [num_frames, num_joints]
    
    # Find valid knee indices
    left_knee_idx, right_knee_idx = find_knee_indices(joint_dof, knee_indices[0], knee_indices[1])
    
    # Extract knee angles for single cycle
    left_knee_angles_single = joint_dof[:, left_knee_idx]  # [num_frames]
    right_knee_angles_single = joint_dof[:, right_knee_idx]  # [num_frames]
    
    # Determine how many frames to generate
    num_frames_single = frames.shape[0]
    dt = 1.0 / fps
    single_cycle_time = (num_frames_single - 1) / fps if num_frames_single > 1 else 0.0
    
    if duration is not None:
        # Calculate number of loops needed for target duration
        num_loops = max(1, int(np.ceil(duration / single_cycle_time)))
        num_frames_target = int(duration * fps)
    elif num_steps is not None:
        # Use exact number of steps
        num_loops = max(1, int(np.ceil(num_steps / num_frames_single)))
        num_frames_target = num_steps
    else:
        # Use num_loops
        num_frames_target = num_loops * num_frames_single
    
    # Repeat the knee angles for the desired number of loops/steps
    if num_loops > 1 or num_steps is not None or duration is not None:
        # Repeat the data
        left_knee_angles = np.tile(left_knee_angles_single, num_loops)[:num_frames_target]
        right_knee_angles = np.tile(right_knee_angles_single, num_loops)[:num_frames_target]
        num_frames = len(left_knee_angles)
    else:
        left_knee_angles = left_knee_angles_single
        right_knee_angles = right_knee_angles_single
        num_frames = num_frames_single
    
    # Create time array
    time = np.arange(num_frames) * dt  # [num_frames]
    
    # Create step/frame number array
    steps = np.arange(num_frames)  # [num_frames]

    print(f"left knee idx:{left_knee_idx}")
    print(f"right knee idx:{right_knee_idx}")
    
    
    return {
        'time': time,
        'steps': steps,
        'left_knee': left_knee_angles,
        'right_knee': right_knee_angles,
        'left_idx': left_knee_idx,
        'right_idx': right_knee_idx,
        'fps': fps,
        'num_frames': num_frames,
        'dt': dt,
        'total_time': time[-1] if len(time) > 0 else 0.0,
        'num_loops': num_loops,
        'single_cycle_frames': num_frames_single,
    }

def plot_knee_angles(knee_data, save_path=None):
    """
    Plot knee angles over time with step numbers.
    
    Args:
        knee_data: Dictionary with 'time', 'steps', 'left_knee', 'right_knee', etc.
        save_path: Optional path to save the plot
    """
    time = knee_data['time']
    steps = knee_data['steps']
    left_knee = knee_data['left_knee']
    right_knee = knee_data['right_knee']
    num_frames = knee_data.get('num_frames', len(time))
    total_time = knee_data.get('total_time', time[-1] if len(time) > 0 else 0.0)
    
    # Filter out NaN values for plotting
    left_valid = ~np.isnan(left_knee)
    right_valid = ~np.isnan(right_knee)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Add overall title with timeframe and step information
    num_loops = knee_data.get('num_loops', 1)
    single_cycle_frames = knee_data.get('single_cycle_frames', num_frames)
    if num_loops > 1:
        title = f'Knee Angles Over Time (Total: {total_time:.3f}s, {num_frames} steps/frames, {num_loops} loops)'
    else:
        title = f'Knee Angles Over Time (Total: {total_time:.3f}s, {num_frames} steps/frames)'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Plot left knee
    if np.any(left_valid):
        axes[0].plot(time[left_valid], left_knee[left_valid], 'b-', linewidth=2, 
                    label=f'Left Knee (L_KFE, idx={knee_data.get("left_idx", "?")})')
        axes[0].set_ylabel('Angle (rad)', fontsize=12)
        axes[0].set_title('Left Knee Angle', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        if np.sum(left_valid) > 0:
            valid_left = left_knee[left_valid]
            axes[0].set_ylim([np.min(valid_left) - 0.2, np.max(valid_left) + 0.2])
        
        # Add secondary x-axis for step numbers
        ax2_left = axes[0].twiny()
        ax2_left.set_xlim(axes[0].get_xlim())
        # Map time ticks to step numbers
        time_ticks = axes[0].get_xticks()
        if len(time) > 1 and len(time_ticks) > 0:
            dt = time[1] - time[0]
            step_ticks = []
            step_labels = []
            for t in time_ticks:
                if 0 <= t <= total_time:
                    step_num = int(t / dt) if dt > 0 else 0
                    if 0 <= step_num < num_frames:
                        step_ticks.append(t)
                        step_labels.append(str(step_num))
            if step_ticks:
                ax2_left.set_xticks(step_ticks)
                ax2_left.set_xticklabels(step_labels, fontsize=9)
        ax2_left.set_xlabel('Step/Frame Number', fontsize=10, color='gray')
        ax2_left.tick_params(axis='x', colors='gray')
    else:
        axes[0].text(0.5, 0.5, 'No valid data for left knee', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Left Knee Angle (No Data)', fontsize=13, fontweight='bold')
    
    # Plot right knee
    if np.any(right_valid):
        axes[1].plot(time[right_valid], right_knee[right_valid], 'r-', linewidth=2, 
                    label=f'Right Knee (R_KFE, idx={knee_data.get("right_idx", "?")})')
        axes[1].set_ylabel('Angle (rad)', fontsize=12)
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_title('Right Knee Angle', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        if np.sum(right_valid) > 0:
            valid_right = right_knee[right_valid]
            axes[1].set_ylim([np.min(valid_right) - 0.2, np.max(valid_right) + 0.2])
        
        # Add secondary x-axis for step numbers
        ax2_right = axes[1].twiny()
        ax2_right.set_xlim(axes[1].get_xlim())
        # Map time ticks to step numbers
        time_ticks = axes[1].get_xticks()
        if len(time) > 1 and len(time_ticks) > 0:
            dt = time[1] - time[0]
            step_ticks = []
            step_labels = []
            for t in time_ticks:
                if 0 <= t <= total_time:
                    step_num = int(t / dt) if dt > 0 else 0
                    if 0 <= step_num < num_frames:
                        step_ticks.append(t)
                        step_labels.append(str(step_num))
            if step_ticks:
                ax2_right.set_xticks(step_ticks)
                ax2_right.set_xticklabels(step_labels, fontsize=9)
        ax2_right.set_xlabel('Step/Frame Number', fontsize=10, color='gray')
        ax2_right.tick_params(axis='x', colors='gray')
    else:
        axes[1].text(0.5, 0.5, 'No valid data for right knee', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Right Knee Angle (No Data)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'knee_angles_plot.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {os.path.join(os.path.dirname(__file__), 'knee_angles_plot.png')}")
    
    plt.show()

def plot_knee_angles_overlay(knee_data, save_path=None):
    """
    Plot both knee angles on the same plot for comparison with step numbers.
    
    Args:
        knee_data: Dictionary with 'time', 'steps', 'left_knee', 'right_knee', etc.
        save_path: Optional path to save the plot
    """
    time = knee_data['time']
    steps = knee_data['steps']
    left_knee = knee_data['left_knee']
    right_knee = knee_data['right_knee']
    num_frames = knee_data.get('num_frames', len(time))
    total_time = knee_data.get('total_time', time[-1] if len(time) > 0 else 0.0)
    
    # Filter out NaN values for plotting
    left_valid = ~np.isnan(left_knee)
    right_valid = ~np.isnan(right_knee)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Add overall title with timeframe and step information
    fig.suptitle(f'Knee Angles Over Time (Total: {total_time:.3f}s, {num_frames} steps/frames)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot both knees
    if np.any(left_valid):
        ax.plot(time[left_valid], left_knee[left_valid], 'b-', linewidth=2, 
                label=f'Left Knee (L_KFE, idx={knee_data.get("left_idx", "?")})', alpha=0.8)
    if np.any(right_valid):
        ax.plot(time[right_valid], right_knee[right_valid], 'r-', linewidth=2, 
                label=f'Right Knee (R_KFE, idx={knee_data.get("right_idx", "?")})', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angle (rad)', fontsize=12)
    ax.set_title('Knee Angles Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Add secondary x-axis for step numbers
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Map time ticks to step numbers
    time_ticks = ax.get_xticks()
    if len(time) > 1 and len(time_ticks) > 0:
        dt = time[1] - time[0]
        step_ticks = []
        step_labels = []
        for t in time_ticks:
            if 0 <= t <= total_time:
                step_num = int(t / dt) if dt > 0 else 0
                if 0 <= step_num < num_frames:
                    step_ticks.append(t)
                    step_labels.append(str(step_num))
        if step_ticks:
            ax2.set_xticks(step_ticks)
            ax2.set_xticklabels(step_labels, fontsize=9)
    ax2.set_xlabel('Step/Frame Number', fontsize=10, color='gray')
    ax2.tick_params(axis='x', colors='gray')
    
    # Set y-axis limits
    all_angles = []
    if np.any(left_valid):
        all_angles.append(left_knee[left_valid])
    if np.any(right_valid):
        all_angles.append(right_knee[right_valid])
    
    if all_angles:
        all_angles = np.concatenate(all_angles)
        ax.set_ylim([np.nanmin(all_angles) - 0.2, np.nanmax(all_angles) + 0.2])
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'knee_angles_overlay.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {os.path.join(os.path.dirname(__file__), 'knee_angles_overlay.png')}")
    
    plt.show()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot knee angles from humanoid_walk.pkl motion file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single cycle (default):
  python plot_knee_angles.py
  
  # Loop motion 5 times:
  python plot_knee_angles.py --num_loops 5
  
  # Generate exactly 500 steps:
  python plot_knee_angles.py --num_steps 500
  
  # Generate 5 seconds of data:
  python plot_knee_angles.py --duration 5.0
  
  # Combine with custom knee indices:
  python plot_knee_angles.py --num_loops 3 --knee_indices 1 17
        """
    )
    
    parser.add_argument(
        '--num_loops',
        type=int,
        default=DEFAULT_NUM_LOOPS,
        help=f'Number of times to repeat/loop the motion (default: {DEFAULT_NUM_LOOPS})'
    )
    
    parser.add_argument(
        '--num_steps',
        type=int,
        default=None,
        help='Exact number of steps/frames to generate (overrides --num_loops)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Target duration in seconds (overrides --num_loops and --num_steps)'
    )
    
    parser.add_argument(
        '--knee_indices',
        type=int,
        nargs=2,
        default=None,
        metavar=('LEFT_IDX', 'RIGHT_IDX'),
        help='Knee joint indices in motion file (will be read from XML if not provided)'
    )
    
    return parser.parse_args()

def main():
    """Main function to load motion data and plot knee angles."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"Loading motion file: {MOTION_FILE_PATH}")
    
    # Load motion data
    motion_data = load_motion_data(MOTION_FILE_PATH)
    
    original_num_frames = motion_data['frames'].shape[0] if hasattr(motion_data['frames'], 'shape') else len(motion_data['frames'])
    
    print(f"Motion data loaded:")
    print(f"  - FPS: {motion_data['fps']}")
    print(f"  - Loop mode: {motion_data.get('loop_mode', 'N/A')}")
    print(f"  - Original number of frames: {original_num_frames}")
    
    # Determine extension method
    if args.duration:
        print(f"\nGenerating data for target duration: {args.duration:.3f} seconds")
    elif args.num_steps:
        print(f"\nGenerating exactly {args.num_steps} steps/frames")
    else:
        print(f"\nLooping motion {args.num_loops} time(s)")
    
    # Get knee indices from XML file if not provided via command line
    if args.knee_indices is None:
        print(f"\nReading knee joint indices from XML file: {HUMANOID_XML_PATH}")
        try:
            left_idx, right_idx = get_knee_indices_from_xml(HUMANOID_XML_PATH)
            args.knee_indices = [left_idx, right_idx]
            print(f"Using indices from XML: {args.knee_indices}")
        except Exception as e:
            print(f"Warning: Could not read indices from XML: {e}")
            print("Falling back to default indices [1, 4]")
            args.knee_indices = [1, 4]
    else:
        print(f"\nUsing provided knee indices: {args.knee_indices}")
    
    # Extract knee angles
    print(f"\nExtracting knee angles using indices: {args.knee_indices}")
    knee_data = extract_knee_angles(
        motion_data, 
        args.knee_indices,
        num_loops=args.num_loops,
        num_steps=args.num_steps,
        duration=args.duration
    )
    
    print(f"Generated data:")
    print(f"  - Total frames: {knee_data['num_frames']}")
    print(f"  - Total time: {knee_data['total_time']:.3f} seconds")
    print(f"  - Number of loops: {knee_data.get('num_loops', 1)}")
    
    print(f"Knee angle statistics (using indices [{knee_data.get('left_idx', '?')}, {knee_data.get('right_idx', '?')}]):")
    
    # Left knee stats
    left_valid = ~np.isnan(knee_data['left_knee'])
    if np.any(left_valid):
        left_angles = knee_data['left_knee'][left_valid]
        print(f"  - Left knee: min={np.min(left_angles):.3f} rad, "
              f"max={np.max(left_angles):.3f} rad, "
              f"mean={np.mean(left_angles):.3f} rad, "
              f"std={np.std(left_angles):.3f} rad")
    else:
        print(f"  - Left knee: ALL NaN values")
    
    # Right knee stats
    right_valid = ~np.isnan(knee_data['right_knee'])
    if np.any(right_valid):
        right_angles = knee_data['right_knee'][right_valid]
        print(f"  - Right knee: min={np.min(right_angles):.3f} rad, "
              f"max={np.max(right_angles):.3f} rad, "
              f"mean={np.mean(right_angles):.3f} rad, "
              f"std={np.std(right_angles):.3f} rad")
    else:
        print(f"  - Right knee: ALL NaN values")
    
    # Check for NaN values
    if np.any(np.isnan(knee_data['left_knee'])) or np.any(np.isnan(knee_data['right_knee'])):
        print("\nWARNING: NaN values detected in knee angles!")
        print(f"  - Left knee NaN count: {np.sum(np.isnan(knee_data['left_knee']))}")
        print(f"  - Right knee NaN count: {np.sum(np.isnan(knee_data['right_knee']))}")
        print("  The script attempted to find alternative indices. If the plot looks wrong,")
        print("  you may need to manually adjust KNEE_JOINT_INDICES in the script.")
    
    # Plot knee angles
    print("\nGenerating plots...")
    plot_knee_angles(knee_data)
    plot_knee_angles_overlay(knee_data)

if __name__ == "__main__":
    main()

