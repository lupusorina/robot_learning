import os
import numpy as np
from datetime import datetime
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
# from robot_learning.src.jax.train import FOLDER_RESULTS
# from robot_learning.src.jax.wrapper import wrap_for_brax_training
from robot_learning.src.jax.wrapper import wrap_for_brax_training
import jax
from robot_learning.src.jax.deepmimic_env import DeepMimicBiped
from robot_learning.src.jax.randomize import domain_randomize

# set up results folder
RESULTS = 'results'
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
FOLDER_RESULTS = os.path.join(RESULTS, time_now)
ABS_FOLDER_RESULTS = os.path.abspath(FOLDER_RESULTS)
os.makedirs(ABS_FOLDER_RESULTS, exist_ok=True)


#debugging deepmimic PPO Config
deepmimic_ppo_config = config_dict.create(
      num_timesteps=1000,  # Small number for debugging
      num_evals=2,  
      reward_scaling=1.0,
      clipping_epsilon=0.2,
      num_resets_per_eval=1,
      episode_length=100,  
      normalize_observations=True,
      action_repeat=1,
      unroll_length=10,  
      num_minibatches=8,  
      num_updates_per_batch=2,  
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=0.005,
      num_envs=128,  
      batch_size=64, 
      max_grad_norm=1.0,
      network_factory=config_dict.create(
        policy_hidden_layer_sizes=(512,256,128),
        value_hidden_layer_sizes=(512,256,128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    ),
    #deepMimic-specific configuration
    imitation_config=config_dict.create(
        enable_imitation=True,
        # where we set out reference motion
        # let's see if this works
        reference_motion_path="/home/marrodri/Documents/code-repositories/robot_learning/robot_learning/src/assets/reference_motions/humanoid_walk.txt",
        pose_scale=10.0,
        velocity_scale=1.0,
        end_effector_scale=5.0,
        orientation_scale=2.0
    ),
  )


#DeepMimic PPO Config
# deepmimic_ppo_config = config_dict.create(
#     num_timesteops = 200_000_000,
#     num_evals=15,
#     reward_scaling=1.0,
#     clipping_epsilon=0.2,
#     num_resets_per_eval=1,
#     episode_length=1000,
#     normalize_observations=True,
#     action_repeat=1,
#     unroll_length=20,
#     num_minibatches=32,
#     num_updates_per_batch=4,
#     discounting=0.97,
#     learning_rate=3e-4,
#     entropy_cost=0.005,
#     num_envs=8192, 
#     batch_size=256,
#     max_grad_norm=1.0,
#     network_factory=config_dict.create(
#         policy_hidden_layer_sizes=(512,256,128),
#         value_hidden_layer_sizes=(512,256,128),
#         policy_obs_key="state",
#         value_obs_key="privileged_state",
#     ),
#     #deepMimic-specific configuration
#     imitation_config=config_dict.create(
#         enable_imitation=True,
#         # where we set out reference motion
#         # let's see if this works
#         reference_motion_path="/home/marrodri/Documents/code-repositories/robot_learning/robot_learning/src/assets/reference_motions/humanoid_walk.txt",
#         pose_scale=10.0,
#         velocity_scale=1.0,
#         end_effector_scale=5.0,
#         orientation_scale=2.0
#     ),
# )

#create DeepMimic environments
env = DeepMimicBiped(
    save_config_folder=ABS_FOLDER_RESULTS,
    reference_motion_path=deepmimic_ppo_config.imitation_config.reference_motion_path,
    imitation_config=deepmimic_ppo_config.imitation_config   
)

eval_env = DeepMimicBiped(
    save_config_folder=ABS_FOLDER_RESULTS,
    refernce_motion_ath=deepmimic_ppo_config.imitation_config.reference_motion_path,
    imitation_config=deepmimic_ppo_config.imitation_config
)

#Progress function
def progress(num_steps, metrics):
    print(f"Step {num_steps}: Reward {metrics.get('eval/episode_reward', 0):.3f}")
    if 'imitation_pose' in metrics:
        print(f" Imitation rewards: pose={metrics.get('imitation_pose', 0):.3f}")

#setup training 
ppo_training_params = dict(deepmimic_ppo_config)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in deepmimic_ppo_config:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **deepmimic_ppo_config.network_factory
    )

#Create training function
train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    randomization_fn=domain_randomize,
    save_checkpoint_path=ABS_FOLDER_RESULTS,
)

#run training
print("Starting DeepMimic training...")
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrap_for_brax_training
)

print("DeepMimic training completed!")