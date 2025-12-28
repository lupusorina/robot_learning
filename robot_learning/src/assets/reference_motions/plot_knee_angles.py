'''
File created by Marthel Rodriguez
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from robot_learning.src.assets.reference_motions.plot_knee_angles_generated import extract_knee_angles

# from robot_learning.src.assets.reference_motions.plot_knee_angles_generated import plot_knee_angles

MOTION_FILE_PATH="/home/marrodri/Documents/code-repositories/robot_learning_sorina/robot_learning/src/assets/reference_motions/humanoid_walk.pkl"

# XML files. But maybe we don't need this
HUMANOID_CHARACTER_FILE_PATH="/home/marrodri/Documents/code-repositories/robot_learning_sorina/robot_learning/src/assets/characters/humanoid.xml"
BIPED_CHARACTER_FILE_PATH=""



# import the pkl file (motion file),
def load_motion_data(file_path):
    with open(file_path, 'rb') as f:
        motion_data = pickle.load(f)
    return motion_data

# and read a single file for parsing
def parse_motion_data(motion_data):
    frames = np.array(motion_data['frames'])
    fps = motion_data['fps']
    frames_durations=[]
    frame_times=[]

    #set the time duration for each frame
    num_frames = frames.shape[0]
    print(f"num_frames:{num_frames}")
    dt = 1.0/fps #this is our actual duration for each frame
    single_cycle_time = (num_frames - 1) / fps if num_frames > 1 else 0.0

    for frame in frames:
        frames_durations.append(dt)
    frames_durations[-1]= 0

    #calculate the acummulated time of each frame 
    frame_times = np.arange(num_frames)*dt

    print(f"frames_duration:{frames_durations}")
    print(f"frames_times:{frame_times}")
    
    return frames, frames_durations, frame_times
    
    pass

# set the loop mode 1.
# and we have the fps set to 120. this can help to get the time duration for each frame.



# import the character file(start with humanoid, 
# then with biped)

'''
the left and right knees values occur on these specific values.

motion frame ids from pickle file to fetch.
-left_knee id:30
-right_knee id: 23
'''

def extract_knee_angles(frames, frame_times, num_loops=20):
    #extract knee_angles
    left_knee_angles=[]
    right_knee_angles=[]
    times = []

    base_duration = frame_times[-1]
    frames = frames[:-1]
    frame_times=frame_times[:-1]


    # print(f"right knee angle: {frames[0][23]}")
    # print(f"left knee angle: {frames[0][30]}")

    for loop_idx in range(num_loops):
        loop_time_offset = loop_idx * base_duration
        for frame_idx, frame in enumerate(frames):
            right_knee_angles.append(frame[23])
            left_knee_angles.append(frame[30])
            times.append(frame_times[frame_idx] + loop_time_offset)

    times.append(num_loops*base_duration)
    # print(f"new times:{times}")

    right_knee_angles = np.degrees(right_knee_angles)
    left_knee_angles = np.degrees(left_knee_angles)
    return np.array(times), np.array(right_knee_angles), np.array(left_knee_angles)

    

# plot the degrees of each knee.
# (the values are going to be in radians, just convert them to degrees)

def plot_knee_angles(frame_times, times, right_knee_angles, left_knee_angles, time_range=20,num_loops=20):
    
    #times array will be replaced with single_cycle_time
    # Filter data if time_range is specified
    
    angle_unit='degrees'
    print(f"right_knee_angles:{right_knee_angles}")
    
    time_range = (0, time_range)
    t_min, t_max = time_range
    mask = (times[:-1] >= t_min) & (times[:-1] <= t_max)
    times_filtered = times[:-1][mask]
    right_angles_filtered = right_knee_angles[mask]
    left_angles_filtered = left_knee_angles[mask]
    # else:
    #     times_filtered = times[:-1]
    #     right_angles_filtered = right_angles
    #     left_angles_filtered = left_angles
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    title = 'Knee Angles Over Time'
    if num_loops > 1:
        title += f' ({num_loops} loops)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot knee angles
    ax.plot(times_filtered, (-1*right_angles_filtered), 'r-', linewidth=2, label='Right Knee', alpha=0.8)
    ax.plot(times_filtered, (-1*left_angles_filtered), 'b-', linewidth=2, label='Left Knee', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(f'Angle ({angle_unit})', fontsize=12)
    ax.set_title(f'Knee Joint Angles ({angle_unit})', fontsize=12)
    
    if time_range is not None:
        ax.set_xlim(time_range)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"Saved plot to {save_path}")
    # else:
    plt.show()
    
    plt.close()
    pass

def main():
    motion_data = load_motion_data(MOTION_FILE_PATH)
    frames, frames_durations, frame_times = parse_motion_data(motion_data)
    times, right_knee_angles, left_knee_angles = extract_knee_angles(frames, frame_times,20)
    plot_knee_angles(frame_times, times, right_knee_angles, left_knee_angles, 3,20)

if __name__ == "__main__":
    main()

