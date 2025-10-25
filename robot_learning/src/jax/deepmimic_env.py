"""Created by Marthel on Oct/24/2025"""

"""Deepmimic-enabled environment wrapper."""

import jax
import jax.numpy as jp
from typing import Optional, Dict, Any
from robot_learning.src.jax import reference_motions
from robot_learning.src.jax.envs.biped import Biped
from robot_learning.src.jax.reference_motions import ReferenceMotionDataset
from robot_learning.src.jax.imitation_rewards import ImitationRewards

class DeepMimicBiped(Biped):
    """Biped environment with DeepMimic imitation capabilities."""

    def __init__(self, reference_motion_path: Optional[str] = None,
                imitation_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)

        # Load reference motion if provided
        if reference_motion_path:
            self.reference_dataset = ReferenceMotionDataset(reference_motion_path)
        else:
            self.reference_dataset = None

        # setup imitation rewards
        if imitation_config:
            self.imitation_rewards = ImitationRewards(imitation_config)
        else:
            self.imitation_rewards = None
    
    def step(self, state, action):
        """Step with DeepMimic imitation rewards."""
        #Call parent step function
        state = super().step(state, action)

        #Add imitation rewards if reference motion exists
        if self.reference_dataset and self.imitation_rewards:
            reference_state = self.reference_dataset.get_reference_state(state.time)
            imitation_rewards = self.imitation_rewards.compute_imitation_rewards(state, reference_state)

            #Add imitation rewards to state metrics
            for key, value in imitation_rewards.items():
                state.metrics[key] = value


        return state