
'''File created by marthel on oct 24/2025'''
"""Deepmimic-style imitation reward functions."""

import jax
import jax.numpy as jp
from typing import Dict, Any

class ImitationRewards:
    """Computes DeepMimic-style imitation rewards."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def compute_imitation_rewards(self, state, reference_state: Dict[str, jp.ndarray]) -> Dict[str, float]:
        """Compute all DeepMimic imitation rewards."""
        rewards ={}

        #1. Pose imitation
        rewards['imitation_pose'] = self.compute_pose_reward(state, reference_state)


        #2. Velocity imitation (joint velocities)
        rewards['imitation_velocity'] = self.compute_velocity_reward(state, reference_state)

        #3. End-effector imitation
        rewards['imitation_end_effector'] = self.compute_end_effector_reward(state, reference_state)


        #4. Orientation imitation
        rewards['imitation_orientation'] = self.compute_orientation_reward(state, reference_state)

        return rewards;

    def compute_pose_reward(self, state, reference_state) -> float:
        """Compute joint position imitation reward."""
        pose_diff = jp.linalg.norm(state.qpos[7:] - reference_state['qpos'][7:])
        return float(jp.exp(-pose_diff * self.config['pose_scale']))

    def compute_velocity_reward(self, state, reference_state) -> float:
        """Compute joint veolicty imitation reward."""
        vel_diff = jp.linalg.norm(state.qvel[6:] - reference_state['qvel'][6:])
        return float(jp.exp(-vel_diff * self.config['velocity_scale']))
    
    def compute_end_effector_reward(self, state, reference_state) -> float:
        """compute end-effector position imitation reward."""
        #this would need to be adapted to your specific end-effector computation
        #another words, this is a placeholder
        ee_diff = jp.linalg.norm(self.get_end_effector_pos(state) - reference_state['ee_pos'])
        return float(jp.exp(-ee_diff*self.config['end_effector_scale']))

    def compute_orientation_reward(self, state, reference_state) -> float:
        """Compute body orientation imitation reward."""
        orientation_diff = jp.linalg.norm(state.qpos[3:7] - reference_state['qpos'][3:7])
        return float(jp.exp(-orientation_diff * self.config['orientation_scale']))

    # TODO: ask for end-effectors and if this code has it.
    def get_end_effector_pos(self, state):
        '''get end-effector positions (adapt to your robot).'''
        # placeholder - implement based on your robot's end-effectors
        return jp.array([0.0, 0.0, 0.0]) #Replace with actual end-effector
    

    
    
