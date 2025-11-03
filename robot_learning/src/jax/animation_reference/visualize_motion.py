
'''

TODO:
1. Load the motion data.
2. Parse the motion data with Json.
3. visualize the motion data with matplotlib.
4. if it works, alter the motion data to use only the lower body and hips.
5. find a way to store the reference motion with its poses, to be used in the training.
QUESTION: IF THE KINEMATIC CHARACTER USES MUJOCO, HOW CAN WE TRANSFORM THE POSES TO THE JAX/MUJOCO ENVIRONMENT?
        (this will be for the future, focus on the small steps first)
'''

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the motion data.
with open('/home/marrodri/Documents/code-repositories/robot_learning_sorina/robot_learning/src/assets/reference_motions/humanoid3d_walk.txt', 'r') as f:
    motion_data = json.load(f)

# Parse the motion data with Json.
print(motion_data)


# class MotionVisualizer:
#     def __init__(self, motion_data):
