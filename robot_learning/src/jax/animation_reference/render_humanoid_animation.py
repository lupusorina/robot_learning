import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
import argparse
from scipy.spatial.transform import Rotation as R


# declaring the characterParser:
class CharacterParser:
    """Parsser for Deepmimic character files"""

    def __init__(self, char_file):
        self.char_file = char_file
        self.joints = []
        self.joint_hierarchy = {}
        self.joint_types = {}
        self.joint_attach_points = {}
        self.num_joints = 0

    def parse(self):
        """Parse the character JSON file"""
        with open(self.char_file, "r") as f:
            data = json.load(f)
        
        skeleton = data.get('Skeleton',{})
        joints_data = skeleton.get("Joints",[])

        #Build joint hierarchy
        for joint_data in joints_data:
            joint_id = joint_data['ID']
            joint_name = joint_data['Name']
            joint_type = joint_data['Type']
            parent_id = joint_data['Parent']

            attach_x = joint_data.get('AttachX', 0.0)
            attach_y = joint_data.get('AttachY', 0.0)
            attach_z = joint_data.get('AttachZ', 0.0)

            self.joints.append({
                'id': joint_id,
                'name': joint_name,
                'type': joint_type,
                'parent': parent_id,
                'attach': np.array([attach_x, attach_y, attach_z])
            })

            self.joint_hierarchy[joint_id] = parent_id
            self.joint_types[joint_id] = joint_type
            self.joint_attach_points[joint_id] = np.array([attach_x, attach_y, attach_z])

        self.num_joints = len(self.joints)
        self.joints.sort(key=lambda x:x['id'])

        print(f"Loaded {self.num_joints} joints from {self.char_file}")
        # constructor finished
        return self


    def get_joint_param_size(self, joint_type):
        """get number of parameter for each joint type"""
        type_sizes = {
            'none': 7, #root: 3 pos + 4 rot
            'spherical': 4, # quaternion
            'revolute': 1, # angle
            'fixed':0, # no params
            'planar': 2,
            'prismatic': 1
        }
        return type_sizes.get(joint_type,0)

    def get_joint_connections(self):
        """get list of parent-child joint connections for skeleton visualization"""
        connections = []
        for joint in self.joints:
            parent_id = joint['parent']
            if parent_id >= 0: #has a parent
                connections.append((parent_id, joint['id']))
        return connections


class MotionParser:
    """Parser for DeepMimic motion files"""

    def __init__(self, motion_file):
        self.motion_file = motion_file
        self.loop_mode = None
        self.frames = []
        self.frames_durations = []
        self.frame_times = []

    def parse(self):
        """Parse the JSON motion file"""
        with open(self.motion_file, 'r') as f:
            data = json.load(f)

        self.loop_mode = data.get("Loop", "none")
        frames_data = data.get('Frames', [])

        for frame in frames_data:
            duration = frame[0]
            pose_data = frame[1:] # 42 DOF: 3(root pos) + 39 (rotations)

            self.frame_durations.append(duration)
            self.frames.append(np.array(pose_data))

        self.frames = np.array(self.frames)

        # calculate cumulative time, why for?
        self.frame_times = np.cumsum([0] + self.frame_durations[-1])
        self.total_duration = sum(self.frame_durations)

        #printing loading status
        print(f"Loaded {len(self.frames)} frames from {self.motion_file}")
        print(f"Total duration: {self.total_duration:.3f} seconds")
        return self

    def get_frame_at_time(self, time):
        """Get interpolated frame at given time"""
        if time < 0:
            time = 0
        if time >= self.total_duration:
            if self.loop_mode == 'wrap':
                # the modulus of time with its total duration
                time = time % self.total_duration
            else:
                return self.frames[-1]
        
        # Find frame indices
        frame_idx = np.searchsorted(self.frame_times, time, side='right') -1
        frame_idx = min(frame_idx, len(self.frames) - 2)

        #Linear interpolation(what does it do?)
        t0 = self.frame_times[frame_idx]
        t1 = self.frame_times[frame_idx + 1] if frame_idx < len(self.frames) - 1 else self.total_duration
        dt = self.frame_durations[frame_idx] if frame_idx < len(self.frame_durations) else 0

        if dt > 0:
            alpha = (time - t0) /dt
            alpha = np.clip(alpha, 0,1)
            frame = (1 - alpha)*self.frames[frame_idx] + alpha * self.frames[frame_idx + 1]
        else:
            frame = self.frames[frame_idx]
        
        return frame


class ForwardKinematics:
    """Compute forward kinematics to get joint world positions"""

    def __init__(self, character):
        self.character = character
    
    def quaternion_to_matrix(self, quat):
        q = quat.copy()

        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.eye(3) # indentify for zero quart.
        q = q / norm

        #Ensure w >= 0 for consistency (quaternion and its negative represent same rotation)
        if q[0] < 0:
            q = -q
        
        w, x, y, z = q
        # from quart to a matrix array
        R = np.array([
            [1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*x), 1-2*(x*x + z*x), 2(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R

        def extract_pose_params(self, pose_data, char_parser):
            """Extract pose parameters for each joint from pose vector"""
            pose_params = {}
            offset = 0

            for joint_id in range(char_parser.num_joints):
                joint = char_parser.joints[joint_id]
                joint_type = joint['type']