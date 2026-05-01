import gymnasium as gym

import torch
import torch.nn as nn
import numpy as np

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer, StepTrainer
from skrl.utils import set_seed
import cleanrl_ppo
from utils import tile_images, save_video, SaveVideoWrapper



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(layer_init(nn.Linear(self.num_observations, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, self.num_actions), std=0.01))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(layer_init(nn.Linear(self.num_observations, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, 1), std=1.0))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["time_limit_bootstrap"] = False # default is False
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


def make_skrl_agent(experiment_name, env, device):
    cfg["experiment"]["experiment_name"] = experiment_name

    memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=False)
    models["value"] = Value(env.observation_space, env.action_space, device)
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    return agent


import os
import subprocess

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
    import tqdm
    import sys
    import datetime
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Name of the experiment")
    parser.add_argument("--eval", type=str, help="checkpoint to evaluate")
    parser.add_argument("--seed", type=int, default=0)
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
        # env = SaveVideoWrapper(env, f"videos/biped_{experiment_name}")
    else:
        env = gym.make_vec(env_name, num_envs=100, vectorization_mode="sync")
    env_not_wrapped = env
    env = wrap_env(env)
    device = env.device

    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    print(f"device: {device}")

    # agent = make_skrl_agent(experiment_name, env, device)
    ppo_args = cleanrl_ppo.Args()
    ppo_args.seed = seed
    agent = cleanrl_ppo.PPO(env, ppo_args, cfg)

    if cli_args.eval:
        agent.set_running_mode("eval")
        agent.load(cli_args.eval)
        eval_timesteps = 1000
        rewards = torch.zeros(eval_timesteps, env.num_envs)
        obs, infos = env_not_wrapped.reset() # reset does nothing for wrapped vectorized envs
        obs = torch.tensor(obs, device=device)
        frames = []
        for i in tqdm.trange(eval_timesteps, desc="Evaluating"):
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
        for iteration in range(1, nb_iterations + 1):
            # Reset domain randomization
            obs, infos = env_not_wrapped.reset() # reset does nothing for wrapped vectorized envs
            obs = torch.tensor(obs, device=device)
            rollout_rewards = torch.zeros((cfg["rollouts"], env.num_envs))
            rollout_benchmark_rewards = torch.zeros((cfg["rollouts"], env.num_envs))
            for i in tqdm.trange(cfg["rollouts"], desc=f"Iteration {iteration}/{nb_iterations}"):
                agent.pre_interaction(timestep=timestep, timesteps=nb_timesteps)
                with torch.no_grad():
                    actions = agent.act(obs, timestep=timestep, timesteps=nb_timesteps)[0]
                    next_obs, rewards, terminated, truncated, infos = env.step(actions)
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
            print(f"Average reward: {average_reward}")
            agent.track_data(f"Reward / Average reward", average_reward)
            agent.track_data(f"Reward / Average benchmark reward", rollout_benchmark_rewards.mean())

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
