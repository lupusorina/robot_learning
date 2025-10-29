'''File created by marthel for our deepmimic playground'''
'''step 1: adding our imports and processing for deepmimic integration'''
import jax
import jax.numpy as jp
import numpy as np
from typing import Dict, Any, Optional
import json

# TODO: check the reference poses returned values, and see where they come from.
class ReferenceMotionDataset:
    def __init(self, motion_file_path):
        #Load reference motion data (e.g., .bhv, .mocap, etc)
        self.motions = self.load_motions(motion_file_path)
        self.current_time = 0.0
    
    '''added file'''
    def load_motion_file(self, file_path: str) -> Dict[str, Any]:
        '''Load reference motion from file (supports .json)'''
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True).item()
        else:
            raise ValueError(f"Unsupported motion file format: {file_path}")

    def get_reference_pose(self, time_step: float)-> Dict[str, jp.ndarray]:
        """ Get refernce state at given time"""
        #interpolate reference motion at given time
        qpos = self.interpolate_qpos(time)
        qvel = self.interpolate_qvel(time)
        ee_pos = self.interpolate_end_effectors(time)

        return {
            'qpos': jp.array(qpos),
            'qvel': jp.array(qvel),
            'ee_pos': jp.array(ee_pos)
            }
    
    def interpolate_qpos(self, time: float) -> np.ndarray:
        """Interpolate joint positions at given time."""
        # implementation depends on your motion data format
        # this is a placeholder, we have to adapt it to our specific motion data
        return self.motion_data['qpos'][int(time *100) % len(self.motion_data['qpos'])]

    def interpolate_qvel(self, time:float) -> np.ndarray:
        """Interpolate joint velocities at given time"""
        return self.motion_data['qvel'][int(time *100) % len(self.motion_data['qvel'])]

    def interpolate_end_effectors(self, time: float) -> np.ndarray:
        '''Interpolate end-effector positions at given time.'''
        return self.motion_data['ee_pos'][int(time*100) % len(self.motion_data['ee_pos'])]
        


    