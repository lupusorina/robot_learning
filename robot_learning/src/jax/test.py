import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
  with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")


try:
  print('Checking that the installation succeeded:')
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".'
  )

print('Installation successful.')

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

import functools

from brax.training.agents.ppo import checkpoint as ppo_checkpoint

import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import mujoco
import numpy as np

import pandas as pd


from etils import epath

RESULTS_FOLDER_PATH = os.path.abspath('results')

# Sort by date and get the latest folder.
folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
print(folders)
numeric_folders = [f for f in folders if f[0].isdigit()]
latest_folder = numeric_folders[-1]
# latest_folder = "20250520-111220"
print(f'Latest folder: {latest_folder}')

# In the latest folder, find the latest folder, ignore the files.
folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
print(folders)

ABS_FOLDER_RESUlTS = epath.Path(RESULTS_FOLDER_PATH) / latest_folder
print(ABS_FOLDER_RESUlTS)

# Tensorboard.
from torch.utils.tensorboard import SummaryWriter

logdir = f"{RESULTS_FOLDER_PATH}/tensorboard_logs/{latest_folder}"
writer = SummaryWriter(log_dir=logdir)

from robot_learning.src.jax.utils import draw_joystick_command

import time
import robot_learning.src.jax.envs.biped as bb

policy_fn_list = []
policy_folder_list = []

USE_LATEST_WEIGHTS = True
if USE_LATEST_WEIGHTS:
  latest_weights_folder = folders[-1]
  print(f'Latest weights folder: {latest_weights_folder}')
  policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)
  policy_fn_list.append(policy_fn)
  policy_folder_list.append(latest_weights_folder)
else:
  for folder in folders:
    policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / folder)
    policy_fn_list.append(policy_fn)
    policy_folder_list.append(folder)

env_name = bb.NAME_ROBOT
print(f'env_name: {env_name}')

for policy_fn, folder in zip(policy_fn_list, policy_folder_list):
  print(f'{folder}')
  config_overrides = {
    "push_config": {
      "enable": False,
      "interval_range": [5.0, 10.0],
      "magnitude_range": [0.05, 1.0],
    },
  }
  eval_env = bb.Biped(config_overrides=config_overrides)
  print(eval_env._mjx_model.body_ipos[1])

  jit_reset = jax.jit(eval_env.reset)
  print(f'JITing reset and step')
  jit_policy = jax.jit(policy_fn)
  step_fn = jax.jit(eval_env.step)
  # step_fn = eval_env.step
  rng = jax.random.PRNGKey(1)

  rollout = []
  modify_scene_fns = []

  x_vel = 0.0  #@param {type: "number"}
  y_vel = 0.0  #@param {type: "number"}
  yaw_vel = 0.0  #@param {type: "number"}
  command = jp.array([x_vel, y_vel, yaw_vel])

  phase_dt = 2 * jp.pi * eval_env.ctrl_dt * 1.5
  phase = jp.array([0, jp.pi])

  state = jit_reset(rng)
  state.info["phase_dt"] = phase_dt
  state.info["phase"] = phase

  # create a df to store the state.metrics data
  metrics_list = []
  ctrl_list = []
  state_list = []
  joints_list = []
  N = 1400
  
  first_quarter = int(N * 0.25)
  second_quarter = int(N * 0.5)
  third_quarter = int(N * 0.75)
  fourth_quarter = N
  
  for i in range(N):
    if i < first_quarter:
      command = jp.array([0.8, 0.0, 0.0]) # Move forward
    elif i < second_quarter:
      command = jp.array([0.0, 0.2, 0.0]) # Move left
    elif i < third_quarter:
      command = jp.array([-0.8, 0.0, 0.0]) # Move back
    else:
      command = jp.array([0.0, -0.2, 0.0]) # Move right
    time_duration = time.time()
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_policy(state.obs, act_rng)
    ctrl_list.append(ctrl)
    state = step_fn(state, ctrl)
    state_list.append(state.obs["state"])
    metrics_list.append(state.metrics)
    if state.done:
      break
    state.info["command"] = command
    rollout.append(state)
    
    joints = state.data.qpos[7:]
    joints_list.append(joints)

    xyz = np.array(state.data.xpos[eval_env._mj_model.body("base_link").id])
    xyz += np.array([0.0, 0.0, 0.0])
    x_axis = state.data.xmat[eval_env._torso_body_id, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    modify_scene_fns.append(
        functools.partial(
            draw_joystick_command,
            cmd=state.info["command"],
            xyz=xyz,
            theta=yaw,
            scl=np.linalg.norm(state.info["command"]),
        )
    )
    time_diff = time.time() - time_duration

  render_every = 1
  fps = 1.0 / eval_env.ctrl_dt / render_every
  print(f"fps: {fps}")
  traj = rollout[::render_every]
  mod_fns = modify_scene_fns[::render_every]

  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[2] = True
  scene_option.geomgroup[3] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

  frames = eval_env.render(
      traj,
      camera="track",
      scene_option=scene_option,
      width=640,
      height=480,
      modify_scene_fns=mod_fns,
  )

  # media.show_video(frames, fps=fps, loop=False)
  media.write_video(f'{ABS_FOLDER_RESUlTS}/joystick_testing_{folder}_xvel_{x_vel}_yvel_{y_vel}_yawvel_{yaw_vel}.mp4', frames, fps=fps)
  print('Video saved')