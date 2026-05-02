# cleanrl ppo implementation adapted to an skrl agent api

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import math
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import skrl.agents.torch
from typing import Any


@dataclass
class Args:
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

def layer_init(layer, bias_const=0.0):
    """LeCun normal: Var(W) = 1/fan_in (fan-in is in_features for Linear)."""
    fan_in = layer.weight.shape[1]
    std = math.sqrt(1.0 / fan_in)
    torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor sees policy ``state``; critic sees ``privileged_state`` (asymmetric PPO) when obs is a Dict."""

    def __init__(self, envs):
        super().__init__()
        ss = envs.observation_space
        if isinstance(ss, gym.spaces.Dict):
            self._priv_dim = int(np.prod(ss["privileged_state"].shape))
            self._policy_dim = int(np.prod(ss["state"].shape))
        else:
            self._priv_dim = None
            self._policy_dim = int(np.prod(ss.shape))
        act_dim = int(np.prod(envs.action_space.shape))
        critic_in = self._priv_dim if self._priv_dim is not None else self._policy_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in, 512)),
            nn.LayerNorm(512),
            nn.SiLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.SiLU(),
            layer_init(nn.Linear(128, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self._policy_dim, 512)),
            nn.LayerNorm(512),
            nn.SiLU(),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.SiLU(),
            layer_init(nn.Linear(128, act_dim)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def _policy_value_obs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._priv_dim is not None:
            # skrl / gymnasium flatten order: sorted keys -> privileged_state, then state
            return x[:, self._priv_dim :], x[:, : self._priv_dim]
        return x, x

    def get_value(self, x):
        _, value_x = self._policy_value_obs(x)
        return self.critic(value_x)

    def get_action_and_value(self, x, action=None):
        # Check for invalid values in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid values in input x. NaN: {torch.isnan(x).sum()}, Inf: {torch.isinf(x).sum()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        policy_x, value_x = self._policy_value_obs(x)
        action_mean = self.actor_mean(policy_x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(value_x)


class PPO(skrl.agents.torch.Agent):
    def __init__(self, env, args, cfg):
        self.args = args
        self.env = env
        self.num_envs = env.num_envs
        self.batch_size = int(env.num_envs * args.num_steps)
        self.minibatch_size = int(self.batch_size // args.num_minibatches)
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"using device: {device}")
        super().__init__({}, device=device, cfg=cfg)

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        
        self.agent = Agent(env).to(device)
        # skrl base agent expects wrapped models to expose a `.device` attribute
        self.agent.device = device
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # skrl checkpoints are built from `checkpoint_modules`; without this, `agent_*.pt` is `{}`.
        self.checkpoint_modules["model"] = self.agent
        self.checkpoint_modules["optimizer"] = self.optimizer

        # ALGO Logic: Storage setup (flattened Dict obs in skrl key order: privileged_state, state)
        ss = env.observation_space
        if isinstance(ss, gym.spaces.Dict):
            flat_obs_dim = int(np.prod(ss["privileged_state"].shape)) + int(np.prod(ss["state"].shape))
        else:
            flat_obs_dim = int(np.prod(ss.shape))
        self._flat_obs_dim = flat_obs_dim
        self.obs = torch.zeros((args.num_steps, env.num_envs, flat_obs_dim)).to(device)
        self.actions = torch.zeros((args.num_steps, env.num_envs) + env.action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, env.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, env.num_envs)).to(device)
        # True failure only; truncation still bootstraps V in GAE.
        self.next_terminated = torch.zeros((args.num_steps, env.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, env.num_envs)).to(device)

        self.global_step = 0
        self.step = 0
        self.iteration = 0

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        next_obs = states
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(next_obs)
            self.values[self.step] = value.flatten()
        self.actions[self.step] = action # todo would be nice to store everyting in record_transition instead
        self.logprobs[self.step] = logprob
        return action, logprob, {}
    
    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )
        self.obs[self.step] = states
        self.next_terminated[self.step] = terminated.flatten().float()
        self.rewards[self.step] = rewards.flatten()
        self.next_obs = next_states


    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self.step += 1
        self.global_step += self.num_envs

        if self.step == self.args.num_steps:
            self.step = 0

            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                iteration = timestep//self.args.num_steps
                nb_iterations = timesteps//self.args.num_steps
                frac = 1.0 - iteration / nb_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
                # print(f"iteration {iteration+1} learning rate: {lrnow}")

            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_terminated[t]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.next_terminated[t]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape(-1, self._flat_obs_dim)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef
                    # print(f"{epoch} {start} loss: {loss.item()}, pg_loss: {pg_loss.item()}, entropy_loss: {entropy_loss.item()}, v_loss: {v_loss.item()}")

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)

            average_reward = self.rewards.mean().item()
            self.writer.add_scalar("charts/average_reward", average_reward, self.global_step)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)


def load_checkpoint(agent: PPO, path: str) -> None:
    """Load weights from an skrl-format dict checkpoint (keys match ``checkpoint_modules``)."""
    agent.load(path)
