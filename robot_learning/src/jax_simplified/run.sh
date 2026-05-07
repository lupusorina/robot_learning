#!/bin/bash

cd robot_learning/src/jax_simplified
MUJOCO_GL=egl \
python skrl_ppo_pytorch.py \
  --train biped \
  --wandb \
  --video \
  --run-name run_name
