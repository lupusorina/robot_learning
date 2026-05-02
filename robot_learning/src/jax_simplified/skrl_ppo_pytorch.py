import os
import tqdm
import torch
import datetime
import argparse
import subprocess
import numpy as np
import gymnasium as gym
import os
import subprocess

import cleanrl_ppo
from skrl.utils import set_seed
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space
import cleanrl_ppo
from utils import tile_images, save_video
import pandas as pd
from logging import TrainingLogger

# Custom imports.
from robot_learning.src.jax_simplified.utils import tile_images, save_video, SaveVideoWrapper
from robot_learning.src.jax_simplified.utils import save_git_info

def observation_to_agent_tensor(obs, observation_space, device):
    """Map reset/step observations to a flat batch tensor (Dict obs: sorted keys, same as skrl / cleanrl_pPO)."""
    if isinstance(obs, torch.Tensor):
        return obs
    return flatten_tensorized_space(tensorize_space(observation_space, obs, device=device))

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs"
cfg["rollouts"] = 1024  # memory_size

VIDEO_INTERVAL_ITERS = 25
VIDEO_DURATION_ITERS = 1
VIDEO_FPS = 25
VIDEO_FRAME_STRIDE = 4

if __name__ == "__main__":
    import sys
    import datetime
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Name of the experiment")
    parser.add_argument("--eval", type=str, help="checkpoint to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--video", action="store_true", help="Record training videos periodically")
    parser.add_argument("--run-name", type=str, default="", help="Optional name appended to the timestamp")
    cli_args = parser.parse_args()
    seed = cli_args.seed
    if cli_args.train:
        experiment_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        run_name = cli_args.run_name.strip().replace(" ", "_")
        if run_name:
            experiment_name = f"{experiment_name}_{run_name}"
    else:
        experiment_name = "eval"
    cfg["experiment"]["experiment_name"] = experiment_name

    # set status bar title
    print(f"\033]2;{experiment_name}\033\\", end="", flush=True)

    set_seed(seed)

    env_name = "biped"
    # env_name = "Pendulum-v1"
    if env_name == "biped":
        import envs.biped as biped
        env = biped.VectorEnv(biped.BipedSim(), num_envs=2048)
    else:
        env = gym.make_vec(env_name, num_envs=100, vectorization_mode="sync")
    env_not_wrapped = env
    env = wrap_env(env)
    device = env.device

    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    print(f"device: {device}")

    ppo_args = cleanrl_ppo.Args()
    ppo_args.seed = seed
    agent = cleanrl_ppo.PPO(env, ppo_args, cfg)

    if cli_args.eval:
        print(f"Loading checkpoint from: {cli_args.eval}")
        agent.set_running_mode("eval")
        cleanrl_ppo.load_checkpoint(agent, cli_args.eval)
        eval_timesteps = 1000
        rewards = torch.zeros(eval_timesteps, env.num_envs)
        obs, infos = env_not_wrapped.reset() # reset does nothing for wrapped vectorized envs
        obs = observation_to_agent_tensor(obs, env.observation_space, device)
        frames = []
        for i in range(eval_timesteps):
            with torch.no_grad():
                actions = agent.act(obs, timestep=i, timesteps=eval_timesteps)[0]
            next_obs, reward, terminated, truncated, infos = env.step(actions)
            frames.append(tile_images(env_not_wrapped.render()))
            rewards[i, :] = reward.flatten()
            obs = next_obs
        print(f"Evaluation reward: {rewards.mean()}")
        save_video(frames, f"{cli_args.eval.replace('.pt', '_eval.mp4')}", fps=1/env.dt)

    if cli_args.train:
        save_git_info(agent.experiment_dir, os.path.dirname(os.path.abspath(__file__)))

        total_timesteps = 1_000_000_000
        # Configure and instantiate the RL trainer.
        cfg_trainer = {"timesteps": total_timesteps//env.num_envs, "headless": True}

        agent.init(trainer_cfg=cfg_trainer)
        nb_timesteps = cfg_trainer["timesteps"]
        nb_iterations = nb_timesteps // cfg["rollouts"]
        agent.set_running_mode("train")
        timestep = 0
        average_reward_list = []
        average_reward_csv = f"runs/{experiment_name}/average_reward.csv"
        os.makedirs(os.path.dirname(average_reward_csv), exist_ok=True)

        training_logger = TrainingLogger(
            use_wandb=bool(cli_args.wandb),
            run_name=experiment_name,
            config={
                "seed": seed,
                "env_name": env_name,
                "rollouts": int(cfg["rollouts"]),
            },
            video_enabled=bool(cli_args.video),
            video_interval_iters=VIDEO_INTERVAL_ITERS,
            video_duration_iters=VIDEO_DURATION_ITERS,
            video_dir=os.path.join(agent.experiment_dir, "videos"),
            video_fps=VIDEO_FPS,
            video_frame_stride=VIDEO_FRAME_STRIDE,
        )

        for iteration in range(1, nb_iterations + 1):
            training_logger.start_iteration(iteration, nb_iterations)
            # Reset domain randomization.
            obs, infos = env_not_wrapped.reset() # reset does nothing for wrapped vectorized envs
            obs = observation_to_agent_tensor(obs, env.observation_space, device)
            rollout_rewards = torch.zeros((cfg["rollouts"], env.num_envs))
            rollout_benchmark_rewards = torch.zeros((cfg["rollouts"], env.num_envs))
            for i in range(cfg["rollouts"]):
                agent.pre_interaction(timestep=timestep, timesteps=nb_timesteps)
                with torch.no_grad():
                    actions = agent.act(obs, timestep=timestep, timesteps=nb_timesteps)[0]
                    next_obs, rewards, terminated, truncated, infos = env.step(actions)
                    training_logger.record_step(infos, terminated, truncated, env_not_wrapped)
                    agent.record_transition(
                        states=obs,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_obs,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos,
                        timestep=timestep,
                        timesteps=nb_timesteps,
                    )
                    rollout_rewards[i,:] = rewards.squeeze()
                    rollout_benchmark_rewards[i,:] = torch.as_tensor(np.array(infos['benchmark_reward'])).squeeze()
                agent.post_interaction(timestep=timestep, timesteps=nb_timesteps)
                obs = next_obs
                timestep += 1
            average_reward = rollout_rewards.mean()
            average_reward_list.append(average_reward)
            # Save to csv file.
            agent.track_data(f"Reward / Average reward", average_reward)
            agent.track_data(f"Reward / Average benchmark reward", rollout_benchmark_rewards.mean())
            training_logger.end_iteration(
                iteration=iteration,
                nb_iterations=nb_iterations,
                timestep=timestep,
                num_envs=env.num_envs,
                average_reward=average_reward,
            )

        # trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)
        # # Fix agents_scope format for single agent (convert from [1] to [(0, num_envs)])
        # trainer.agents_scope = [(0, env.num_envs)]
        # print(trainer.agents_scope)
        # print(trainer.num_simultaneous_agents)
        # print(trainer.agents)
        # for timestep in range(cfg_trainer["timesteps"]):
        #     trainer.train(timestep=timestep)
        #     trainer.agents = agent # this keeps getting overwritten to a list (skrl bug)
        #     if timestep % 1024 == 0:
        #         env_not_wrapped.reset()
