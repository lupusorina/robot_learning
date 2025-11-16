"""
  Joystick task for the Caltech Biped.
  Modified by Sorina Lupu (eslupu@caltech.edu) from the Berkeley biped code from MuJoCo playground
  https://github.com/google-deepmind/mujoco_playground/
"""

from typing import Any, Dict, Optional, Union, Sequence
import argparse
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
import mujoco.viewer
from etils import epath
from tqdm import tqdm

# Local imports.
import robot_learning.src.jax.utils as utils
from robot_learning.src.jax.utils import geoms_colliding, draw_joystick_command
import robot_learning.src.jax.mjx_env as mjx_env

import functools

import os
import json
import shutil

import time
import threading

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: pynput not available. Keyboard input disabled.")
    print("Install with: pip install pynput")

from biped import Biped

# Global set to track currently pressed keys
_pressed_keys = set()
_pressed_keys_lock = threading.Lock()

def on_press(key):
    """Callback for when a key is pressed."""
    try:
        with _pressed_keys_lock:
            _pressed_keys.add(key)
    except:
        pass

def on_release(key):
    """Callback for when a key is released."""
    try:
        with _pressed_keys_lock:
            _pressed_keys.discard(key)
    except:
        pass

def keyboard_input_thread(command_dict, command_lock, MAX_X_VEL, MAX_Y_VEL, MAX_YAW_VEL, running_flag):
    """Thread function to continuously read keyboard input."""
    if not KEYBOARD_AVAILABLE:
        return
    
    # Start keyboard listener in this thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    try:
        while running_flag[0]:
            # Initialize command values
            x_vel = 0.0
            y_vel = 0.0
            yaw_vel = 0.0
            
            # Read currently pressed keys
            with _pressed_keys_lock:
                pressed = _pressed_keys.copy()
            
            # Arrow keys for forward/backward and rotation
            for key in pressed:
                # Arrow keys: RIGHT/LEFT control y_vel, UP/DOWN control x_vel
                if key == keyboard.Key.right:
                    y_vel = -MAX_Y_VEL
                elif key == keyboard.Key.left:
                    y_vel = MAX_Y_VEL
                elif key == keyboard.Key.up:
                    x_vel = MAX_X_VEL
                elif key == keyboard.Key.down:
                    x_vel = -MAX_X_VEL
                # 'w'/'s' control yaw (w = positive, s = negative)
                elif hasattr(key, 'char'):
                    if key.char == 'w':
                        yaw_vel = MAX_YAW_VEL
                    elif key.char == 's':
                        yaw_vel = -MAX_YAW_VEL
            
            # Update shared command dictionary
            with command_lock:
                command_dict['x_vel'] = x_vel
                command_dict['y_vel'] = y_vel
                command_dict['yaw_vel'] = yaw_vel
            
            time.sleep(0.01)  # Small sleep to prevent busy-waiting
    finally:
        listener.stop()

def test_joystick_command() -> None:
  import mediapy as media

  jax.config.update('jax_debug_nans', True)

  # Load Policy.
  RESULTS_FOLDER_PATH = os.path.abspath('../results')
  print(f'RESULTS_FOLDER_PATH: {RESULTS_FOLDER_PATH}')
  folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
  numeric_folders = [f for f in folders if f[0].isdigit()]
  latest_folder = numeric_folders[-1]
  print(f'Latest folder with trained policy: {latest_folder}')

  # In the latest folder, find the latest folder, ignore the files.
  folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
  folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]
  latest_weights_folder = folders[-1]
  print(f'         latest weights folder: {latest_weights_folder}')

  policy_fn_list = []
  policy_folder_list = []

  policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)
  policy_fn_list.append(policy_fn)
  policy_folder_list.append(latest_weights_folder)

  for policy_fn, folder in zip(policy_fn_list, policy_folder_list):
    print(f'{folder}')
    config_overrides = {
      "push_config": {
        "enable": False,
        "interval_range": [5.0, 10.0],
        "magnitude_range": [0.05, 1.0],
      },
    }
    eval_env = Biped(config_overrides=config_overrides)

    # Create a separate MuJoCo model for visualization
    viz_model, viz_data = eval_env.create_mujoco_viz_model()

    jit_reset = jax.jit(eval_env.reset)
    print(f'JITing reset and step')
    jit_policy = jax.jit(policy_fn)
    step_fn = jax.jit(eval_env.step)
    # step_fn = eval_env.step
    rng = jax.random.PRNGKey(1)

    rollout = []

    # Command scaling factors
    MAX_X_VEL = 0.2  # m/s
    MAX_Y_VEL = 0.2  # m/s
    MAX_YAW_VEL = 0.2  # rad/s

    # Initialize keyboard input thread if available
    command_dict = {'x_vel': 0.0, 'y_vel': 0.0, 'yaw_vel': 0.0}
    command_lock = threading.Lock()
    running_flag = [True]  # Use list to allow modification from thread
    
    keyboard_thread = None
    print(f'KEYBOARD_AVAILABLE: {KEYBOARD_AVAILABLE}')
    if KEYBOARD_AVAILABLE:
        # Start the keyboard input thread for interactive control if keyboard is available
        print("Keyboard input enabled. Use the following keys to control the robot:")
        print("  ↑/↓        : Forward / Backward")
        print("  ←/→        : Left / Right")
        print("  W/S        : Rotate Left / Rotate Right")
        keyboard_thread = threading.Thread(
            target=keyboard_input_thread,
            args=(command_dict, command_lock, MAX_X_VEL, MAX_Y_VEL, MAX_YAW_VEL, running_flag),
            daemon=True
        )
        keyboard_thread.start()

    x_vel = 0.0  #@param {type: "number"}
    y_vel = 0.0  #@param {type: "number"}
    yaw_vel = 0.0  #@param {type: "number"}
    command = jp.array([x_vel, y_vel, yaw_vel])

    phase_dt = 2 * jp.pi * eval_env.ctrl_dt * 1.5
    phase = jp.array([0, jp.pi])

    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase

    try:
      with mujoco.viewer.launch_passive(viz_model, viz_data) as viewer:

        while True:
          # Read keyboard input from shared dictionary if available
          if KEYBOARD_AVAILABLE:
              with command_lock:
                  x_vel = command_dict['x_vel']
                  y_vel = command_dict['y_vel']
                  yaw_vel = command_dict['yaw_vel']
              command = jp.array([x_vel, y_vel, yaw_vel])
          else:
              # Fallback: use sampled command if keyboard not available.
              command = eval_env.sample_command(rng)

          time_duration = time.time()
          act_rng, rng = jax.random.split(rng)
          ctrl, _ = jit_policy(state.obs, act_rng)
          state = step_fn(state, ctrl)

          state.info["command"] = command

          # Update MuJoCo visualization with current JAX state.
          viz_data = eval_env.update_mujoco_viz_from_jax_state(viz_model, viz_data, state)
          viewer.sync()

    finally:
      # Clean up: stop the keyboard thread
      if keyboard_thread is not None:
          running_flag[0] = False
          keyboard_thread.join(timeout=1.0)

def main():
  test_joystick_command()

if __name__ == '__main__':
  main()