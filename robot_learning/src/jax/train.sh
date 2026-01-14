#! /bin/bash

# Memory management for CUDA/JAX
# Reduce GPU memory usage to prevent OOM errors
export XLA_CLIENT_MEM_FRACTION=0.7

# Alternative: Disable command buffers (uncomment if still getting OOM)
# export XLA_FLAGS='--xla_gpu_enable_command_buffer='

#MUJOCO_GL=egl python3 train.py
#echo "Script 1. MUJOCO_GL=egl python3 train.py Done."

MUJOCO_GL=egl python3 train_with_motion.py
