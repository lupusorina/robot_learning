import gymnasium as gym
import os
import subprocess

import torch
import torch.nn as nn
import numpy as np

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space
import cleanrl_ppo
from utils import tile_images, save_video
import pandas as pd
from logging import TrainingLogger


def observation_to_agent_tensor(obs, observation_space, device):
    """Map reset/step observations to a flat batch tensor (Dict obs: sorted keys, same as skrl / cleanrl_pPO)."""
    if isinstance(obs, torch.Tensor):
        return obs
    return flatten_tensorized_space(tensorize_space(observation_space, obs, device=device))

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["time_limit_bootstrap"] = True
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["mixed_precision"] = False
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/biped"

VIDEO_INTERVAL_ITERS = 50
VIDEO_DURATION_ITERS = 1
VIDEO_FPS = 25
VIDEO_FRAME_STRIDE = 4


def _try_run_git_command(args, cwd):
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def save_git_info(output_dir, cwd):
    os.makedirs(output_dir, exist_ok=True)
    # Save git hash
    git_hash = _try_run_git_command(['git', 'rev-parse', 'HEAD'], cwd)
    with open(os.path.join(output_dir, "git_hash.txt"), "w") as f:
        f.write(git_hash + "\n")
    # Save branch name
    git_branch = _try_run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd)
    with open(os.path.join(output_dir, "git_branch.txt"), "w") as f:
        f.write(git_branch + "\n")
    # Save diffs
    git_diff = _try_run_git_command(['git', 'diff'], cwd)
    with open(os.path.join(output_dir, "git_diff.patch"), "w") as f:
        f.write(git_diff)



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
    cli_args = parser.parse_args()
    seed = cli_args.seed
    if cli_args.train:
        experiment_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + f"_seed{seed}" + "_" + cli_args.train.replace(" ", "_")
    else:
        experiment_name = "eval"

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
        agent.load(cli_args.eval)
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
        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": total_timesteps//env.num_envs, "headless": True}
        # trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
        # print(trainer.num_simultaneous_agents)
        # print(trainer.agents)
        # print(trainer.agents_scope)
        # # start training
        # trainer.train()

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
            pd.DataFrame(average_reward_list,
                         columns=["average_reward"]).to_csv(average_reward_csv, index=False)
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
