#!/bin/bash

# Match ``robot_learning/src/jax/train.py`` Brax batching (batch_size=256, num_minibatches=32).
# SKRL ``rollouts`` = unroll_length * (256*32 // num_envs); with 8192 envs and unroll 20 -> rollouts 20.

export MUJOCO_GL=egl; python skrl_ppo_pytorch.py  --train biped
