import gymnasium as gym

import inspect
import torch
import torch.nn as nn
import numpy as np
import dataclasses

from skrl.agents.torch.ppo import PPO
try:
    # Older skrl releases
    from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG as _PPO_DEFAULTS
except ImportError:
    try:
        # Some skrl releases renamed this symbol
        from skrl.agents.torch.ppo import PPO_DEFAULT_CFG as _PPO_DEFAULTS
    except ImportError:
        # Newer skrl releases expose a dataclass config type only
        from skrl.agents.torch.ppo import PPO_CFG as _PPO_DEFAULTS
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
try:
    from skrl.resources.schedulers.torch import KLAdaptiveRL
except ImportError:
    from skrl.resources.schedulers.torch import KLAdaptiveLR as KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer, StepTrainer
try:
    from skrl.trainers.torch import TrainerCfg
except ImportError:
    TrainerCfg = None
from skrl.utils import set_seed
from envs.biped import tile_images, save_video



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _features_from_model_inputs(inputs: dict) -> torch.Tensor:
    """SKRL passes env features in ``observations``; ``states`` may be None when there is no separate state."""
    x = inputs.get("states")
    if x is not None:
        return x
    x = inputs.get("observations")
    if x is not None:
        return x
    raise KeyError("Model inputs must include 'observations' or non-null 'states'")


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(layer_init(nn.Linear(self.num_observations, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, self.num_actions), std=0.01))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(_features_from_model_inputs(inputs)), {"log_std": self.log_std_parameter}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(layer_init(nn.Linear(self.num_observations, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, 64)),
                                 nn.LayerNorm(64),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(64, 1), std=1.0))

    def compute(self, inputs, role):
        return self.net(_features_from_model_inputs(inputs)), {}


cfg = dataclasses.asdict(_PPO_DEFAULTS()) if callable(_PPO_DEFAULTS) else _PPO_DEFAULTS.copy()

cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["time_limit_bootstrap"] = False # default is False
if "gae_lambda" in cfg:
    cfg["gae_lambda"] = 0.95
else:
    cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
# cfg["learning_rate_scheduler"] = KLAdaptiveRL
# cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
# Only available in some skrl versions
if "clip_predicted_values" in cfg:
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
cfg.setdefault("experiment", {})
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

def set_agent_mode(agent, mode: str):
    """Compatibility wrapper for skrl running mode API changes."""
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode(mode)
        return
    if hasattr(agent, "enable_training_mode"):
        agent.enable_training_mode(mode == "train", apply_to_models=True)
        return
    if hasattr(agent, "enable_models_training_mode"):
        agent.enable_models_training_mode(mode == "train")
        return
    raise AttributeError("Could not set agent mode: unsupported skrl API")

def agent_act_compat(agent, obs, timestep: int, timesteps: int):
    """Call PPO.act with the correct signature for this skrl version (do not use broad except TypeError)."""
    params = list(inspect.signature(agent.act).parameters)
    if "states" in params:
        return agent.act(obs, None, timestep=timestep, timesteps=timesteps)
    return agent.act(obs, timestep=timestep, timesteps=timesteps)


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
    args = parser.parse_args()
    seed = args.seed
    if args.train:
        experiment_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + f"_seed{seed}" + "_" + args.train.replace(" ", "_")
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
        env = biped.SaveVideoWrapper(env, f"videos/biped_{experiment_name}")
    else:
        env = gym.make_vec(env_name, num_envs=100, vectorization_mode="sync")
    env_not_wrapped = env
    # env.device = "cpu"
    env = wrap_env(env)
    device = env.device

    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    print(f"device: {device}")

    agent = make_skrl_agent(experiment_name, env, device)
    # args = cleanrl_ppo.Args()
    # args.seed = seed
    # agent = cleanrl_ppo.PPO(env, args, cfg)


    if args.eval:
        set_agent_mode(agent, "eval")
        agent.load(args.eval)
        eval_timesteps = 1000
        rewards = torch.zeros(eval_timesteps, env.num_envs)
        obs, infos = env_not_wrapped.reset() # reset does nothing for wrapped vectorized envs
        obs = torch.tensor(obs, device=device)
        frames = []
        for i in tqdm.trange(eval_timesteps, desc="Evaluating"):
            with torch.no_grad():
                actions = agent_act_compat(agent, obs, timestep=i, timesteps=eval_timesteps)[0]
            next_obs, reward, terminated, truncated, infos = env.step(actions)
            frames.append(tile_images(env_not_wrapped.render()))
            rewards[i, :] = reward.flatten()
            obs = next_obs
        print(f"Evaluation reward: {rewards.mean()}")
        save_video(frames, f"{args.eval.replace('.pt', '_eval.mp4')}", fps=1/env.dt)

    if args.train:
        save_git_info(agent.experiment_dir, os.path.dirname(os.path.abspath(__file__)))

        total_timesteps = 200_000_000
        # configure and instantiate the RL trainer
        cfg_trainer_dict = {"timesteps": total_timesteps//env.num_envs, "headless": True}
        cfg_trainer = TrainerCfg(**cfg_trainer_dict) if TrainerCfg is not None else cfg_trainer_dict
        # trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
        # print(trainer.num_simultaneous_agents)
        # print(trainer.agents)
        # print(trainer.agents_scope)
        # # start training
        # trainer.train()

        agent.init(trainer_cfg=cfg_trainer)
        nb_timesteps = cfg_trainer_dict["timesteps"]
        nb_iterations = nb_timesteps // cfg["rollouts"]
        set_agent_mode(agent, "train")
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
                    actions = agent_act_compat(agent, obs, timestep=timestep, timesteps=nb_timesteps)[0]
                    next_obs, rewards, terminated, truncated, infos = env.step(actions)
                    agent.record_transition(
                        observations=obs,
                        states=obs,
                        actions=actions,
                        rewards=rewards,
                        next_observations=next_obs,
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

