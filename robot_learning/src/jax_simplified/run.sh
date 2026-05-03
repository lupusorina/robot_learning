#!/bin/bash

if [ "$1" == "learn_ppo_policy" ]; then
    CUDA_VISIBLE_DEVICES=1; export MUJOCO_GL=egl; python skrl_ppo_pytorch.py  --train biped
fi


if [ "$1" == "learn_online" ]; then
    export MUJOCO_GL=egl; CUDA_VISIBLE_DEVICES=1  python3 online_learning/cleanrl_sac.py \
        --render_mode rgb_array \
        --env_id Biped \
        --runs_directory runs_biped \
        --exp_name trial_1 \
        --num_envs 1 \
	--capture_video
fi
