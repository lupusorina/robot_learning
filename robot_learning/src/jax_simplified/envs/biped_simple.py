import os
import sys
import time
import torch
import mujoco
import numpy as np
from torch import nn
import gymnasium as gym
from gymnasium import spaces
import scipy.spatial.transform as transform
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import robot_learning.src.assets.biped.config as robot_config

# Import custom modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embodied_ant_env')))

# Constants.
WORKSPACE_LENGTH = 10.0 # m


class _ScalarRingBuffer:
    """Rolling window of scalar samples; ``sum()`` is over the last ``n`` pushes (partial until filled)."""

    __slots__ = ("buf", "count", "idx", "n")

    def __init__(self, n: int):
        self.n = n
        self.buf = np.zeros(n, dtype=np.float64)
        self.idx = np.int32(0)
        self.count = 0

    def reset(self) -> None:
        self.buf.fill(0.0)
        self.idx = np.int32(0)
        self.count = 0

    def push(self, x: float) -> None:
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.n
        self.count = min(self.count + 1, self.n)

    def sum(self) -> float:
        if self.count < self.n:
            return float(np.sum(self.buf[: self.count]))
        return float(np.sum(self.buf))


class _NumpyObsRingBuffer:
    """Rolling window over flattened observation vectors (same indexing as utils.RingBuffer)."""

    __slots__ = ("buf", "idx", "n")

    def __init__(self, frame: np.ndarray, n: int):
        f = np.asarray(frame, dtype=np.float32).reshape(-1)
        self.n = n
        self.buf = np.tile(f, (n, 1))
        self.idx = np.int32(0)

    def push(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, dtype=np.float32).reshape(-1)
        self.buf[self.idx] = f
        self.idx = (self.idx + 1) % self.n

    def stacked_most_recent_first(self) -> np.ndarray:
        k = np.arange(self.n, dtype=np.int32)
        i = (self.idx - 1 - k) % self.n
        hist = self.buf[i].reshape(-1)
        return hist[::-1].astype(np.float32)


class EnergyTask:
    """Reward minimizes knee motor effort (squared torques on L_KFE / R_KFE)."""

    def __init__(
        self,
        knee_torque_cost_scale: float = 2.5e-4,
        control_dt: float = 0.01,
        energy_window_duration: float | None = 0.5,
    ):
        self.last_delta_action = np.zeros(2)
        self.knee_torque_cost_scale = knee_torque_cost_scale
        self.control_dt = control_dt
        # Integrated squared knee torque over the last ``energy_window_duration`` seconds
        # (sum of per-step τ²·Δt). None disables ring buffering.
        if energy_window_duration is None or energy_window_duration <= 0:
            self._energy_ring: _ScalarRingBuffer | None = None
        else:
            n = max(1, int(round(energy_window_duration / control_dt)))
            self._energy_ring = _ScalarRingBuffer(n)
        # Single-frame dim: linvel(3)+gyro(3)+up(3)+cmd(3)+q(8)+qd(8)+tau(8)+phase(4)=40.
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(40,),
                                            dtype=np.float32)

    @property
    def knee_torque_energy_in_window(self) -> float:
        """∑ τ²·Δt over the last window (same τ² units as the reward, scaled by Δt)."""
        if self._energy_ring is None:
            return 0.0
        return self._energy_ring.sum()

    def reset(self, info, delta_action=np.zeros(2)):
        self.last_delta_action = delta_action
        if self._energy_ring is not None:
            self._energy_ring.reset()
        return self(info, delta_action)

    def __call__(self, info, delta_action):
        self.last_delta_action = delta_action.copy()
        terminated = False
        truncated = False

        torques = info["torques"]
        idx = info["knee_actuator_indices"]
        tau_knee = torques[idx]
        if self._energy_ring is not None:
            step_energy = float(np.sum(np.square(tau_knee)) * self.control_dt)
            self._energy_ring.push(step_energy)
            info["knee_torque_energy_in_window"] = self.knee_torque_energy_in_window
        # Penalize squared torque (smooth proxy for motor electrical / heating loss).
        reward = -self.knee_torque_cost_scale * np.sum(np.square(tau_knee))
        
        info["original_reward"] = reward

        observation = np.concatenate([
                info["linvel"], # 3
                info["gyro"], # 3
                info["up_B"], # 3
                info["command"], # 3
                info["q_joints"] - info["default_q_joints"], # 8 
                info["q_vel"], # 8
                torques, # 8
                info["phase"], # 4 (cos/sin for two gait phases)
            ], axis=None)
        return observation, reward, terminated, truncated



class BipedSimpleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    def __init__(
        self,
        model_path: str = robot_config.XML_PATH,
        render_mode: str | None = None,
        control_dt: float = 0.01,
        task: EnergyTask | None = None,
        energy_window_duration: float | None = 0.5,
        base_policy: nn.Module = None,
        history_len: int = 3,
        terminate_on_upside_down: bool = True,
    ):
        super().__init__()
        # Initialize the environment.
        sim_dt = 0.001
        self.dt = control_dt
        self.nb_sim_per_step = int(control_dt / sim_dt)
        self.history_len = history_len

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32) # Just knee right and knee left.

        self.task = task if task is not None else EnergyTask(
            control_dt=control_dt,
            energy_window_duration=energy_window_duration,
        )
        single_obs_dim = int(self.task.observation_space.shape[0])
        self.observation_space = spaces.Box(
            low=self.task.observation_space.low[0],
            high=self.task.observation_space.high[0],
            shape=(single_obs_dim * history_len,),
            dtype=np.float32,
        )

        # Initialize the renderer.
        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            width=640,
            height=480,
            max_geom=1000,
            visual_options={},
        )

        # Base policy.
        self.base_policy = base_policy
        
        # Initialize the starting state.
        self._init_q = np.array(self.model.keyframe("home").qpos)
        self._default_q_joints = np.array(self.model.keyframe("home").qpos[7:])
        self._terminate_on_upside_down = terminate_on_upside_down

       # Mapping from joint names to the PPO action indices.
        self.idx_actuators_dict = {}
        for i in range(0, self.model.nu):
            self.idx_actuators_dict[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] = i

        self.actuated_joint_names_to_policy_idx_dict = {
            "L_HAA": 0,
            "L_HFE": 1,
            "L_KFE": 2,
            "R_HAA": 3,
            "R_HFE": 4,
            "R_KFE": 5,
            }
        for name in self.actuated_joint_names_to_policy_idx_dict:
            assert name in self.idx_actuators_dict, f"{name} is not in {self.idx_actuators_dict.keys()}"

        # MuJoCo actuator mapping.
        self.joint_names_to_actuator_idx_dict = { name: int(self.model.joint(name).qposadr[0] - 7) \
                                                for name in self.idx_actuators_dict.keys() }
        self.policy_idx_to_mujoco_actuator_idx_dict = { self.actuated_joint_names_to_policy_idx_dict[name]: self.joint_names_to_actuator_idx_dict[name] 
                                    for name in self.actuated_joint_names_to_policy_idx_dict }

        self._knee_actuator_names = tuple(
            f"{side}_{jn}"
            for side in robot_config.SIDES
            for jn in robot_config.KNEE_JOINT_NAMES
        )
        self._knee_actuator_indices = np.array(
            [self.idx_actuators_dict[n] for n in self._knee_actuator_names],
            dtype=np.int32,
        )

        # Initialize the sensor adresses.
        self._sensor_adr = {}
        for name in [
                    robot_config.GRAVITY_SENSOR,
                    robot_config.LOCAL_LINVEL_SENSOR,
                    robot_config.GYRO_SENSOR,
                    robot_config.GLOBAL_LINVEL_SENSOR,
                    robot_config.GLOBAL_ANGVEL_SENSOR,
                    robot_config.ACCELEROMETER_SENSOR,
                ]:
            sid = self.model.sensor(name).id
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            self._sensor_adr[name] = (adr, dim)

        # Phase for the gait.
        gait_freq = 1.5
        phase_dt = 2 * np.pi * self.dt * gait_freq
        phase = np.array([0, np.pi])

        self.info = {
            'phase': phase,
            'phase_dt': phase_dt,
        }
        self.last_act = np.zeros(8)
        self._obs_history: _NumpyObsRingBuffer | None = None
        self._ppo_policy_obs_history: _NumpyObsRingBuffer | None = None


    def compute_current_obs_for_the_ppo_policy(self, command: np.ndarray) -> tuple[np.ndarray, dict]:
        ''' Compute the current observation. Should be feasible on hardware. '''
        linvel = self._get_sensor_data(robot_config.LOCAL_LINVEL_SENSOR)
        gyro = self._get_sensor_data(robot_config.GYRO_SENSOR)
        up_B = self._get_sensor_data(robot_config.GRAVITY_SENSOR)
        q_joints = self.data.qpos[7:]
        q_vel = self.data.qvel[6:]

        # Phase.
        phase_tp1 = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        cos = np.cos(self.info["phase"]) # cos(phase)
        sin = np.sin(self.info["phase"]) # sin(phase)
        phase = np.concatenate([cos, sin]) # [cos(phase), sin(phase)]

        info = {}
        info["linvel"] = linvel
        info["gyro"] = gyro
        info["up_B"] = up_B
        info["command"] = command
        info["q_joints"] = q_joints
        info["default_q_joints"] = self._default_q_joints
        info["q_vel"] = q_vel
        info["last_act"] = self.last_act
        info["phase"] = phase

        return np.concatenate([
            linvel,
            gyro,
            up_B,
            command,
            q_joints - self._default_q_joints,
            q_vel,
            self.last_act,
            phase,
        ]).clip(-10.0, 10.0), info

    def _stacked_obs_from_buffer(self) -> np.ndarray:
        """Match biped.py build_obs: concatenate history frames, most recent first."""
        return self._obs_history.stacked_most_recent_first()

    def _stacked_ppo_policy_obs_from_buffer(self) -> np.ndarray:
        return self._ppo_policy_obs_history.stacked_most_recent_first()

    def step(self, delta_action: np.ndarray):

        # Get the PPO policy input.
        command = np.zeros(3)
        obs_for_ppo_policy, _ = self.compute_current_obs_for_the_ppo_policy(command=command)
        if self._ppo_policy_obs_history is None:
            self._ppo_policy_obs_history = _NumpyObsRingBuffer(obs_for_ppo_policy, self.history_len)
        else:
            self._ppo_policy_obs_history.push(obs_for_ppo_policy)
        ppo_policy_input = self._stacked_ppo_policy_obs_from_buffer()
        ppo_policy_input_torch = torch.from_numpy(ppo_policy_input).float().reshape(1, -1)
        ppo_action = self.base_policy.get_action(ppo_policy_input_torch).detach().numpy().flatten()

        action_complete = np.zeros(self.model.nu)
        for _, policy_idx in self.actuated_joint_names_to_policy_idx_dict.items():
            if policy_idx is None:
                continue
            action_complete[self.policy_idx_to_mujoco_actuator_idx_dict[policy_idx]] = ppo_action[policy_idx]
    
        # Add the delta actions from the online learning policy.
        delta_action = np.clip(delta_action, self.action_space.low, self.action_space.high)
        for i, name in enumerate(self._knee_actuator_names):
            action_complete[self.idx_actuators_dict[name]] += delta_action[i]

        motor_targets = self._default_q_joints + action_complete

        self.data.ctrl[:] = motor_targets

        mujoco.mj_step(self.model, self.data, nstep=self.nb_sim_per_step)
        mujoco.mj_rnePostConstraint(self.model, self.data) # See https://github.com/openai/gym/issues/1541

        # Get observation and reward from task.
        info = self.get_observation()
        obs_single, reward, terminated, truncated = self.task(info, delta_action)
        if self._obs_history is None:
            self._obs_history = _NumpyObsRingBuffer(obs_single, self.history_len)
        else:
            self._obs_history.push(obs_single)
        observation = self._stacked_obs_from_buffer()

        # Check if out of bounds or nans or truncated from task.
        truncated = self._get_truncated_out_of_bounds_or_nans() or truncated

        # Terminate on upside down.
        quaternion_wxyz = self.data.qpos[3:7]
        up_vector_ant_in_world = transform.Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix()[:, 2]
        z_world = np.array([0, 0, 1])
        upside_down = np.dot(up_vector_ant_in_world, z_world)
        if self._terminate_on_upside_down == True:
            terminated = upside_down < 0
        else:
            terminated = False

        # Render.
        if self.render_mode == "human":
            # Add an arrow to the scene
            self.render()
            
        self.last_act = ppo_action.copy()

        return observation, reward, terminated, truncated, info

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        self.data.ctrl[:] = qpos[7:]
        mujoco.mj_step(self.model, self.data, nstep=self.nb_sim_per_step)
        mujoco.mj_rnePostConstraint(self.model, self.data) # See https://github.com/openai/gym/issues/1541

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def _get_truncated_out_of_bounds_or_nans(self):
        truncation_condition = (
            np.isnan(self.data.qpos).any() | np.isnan(self.data.qvel).any() |
            (self.data.qpos[0] < -WORKSPACE_LENGTH / 2.0) | (self.data.qpos[0] > WORKSPACE_LENGTH / 2.0) |
            (self.data.qpos[1] < -WORKSPACE_LENGTH / 2.0) | (self.data.qpos[1] > WORKSPACE_LENGTH / 2.0)
        )

        return bool(truncation_condition)

    def _get_sensor_data(self, sensor_name: str) -> np.ndarray:
        ''' Get the sensor data. '''
        adr, dim = self._sensor_adr[sensor_name]
        return self.data.sensordata[adr:adr + dim]

    def get_observation(self):
        ''' Same as the PPO policy input for now '''
        _, info = self.compute_current_obs_for_the_ppo_policy(command=np.zeros(3))
        info["torques"] = self.data.actuator_force
        info["knee_actuator_indices"] = self._knee_actuator_indices
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.step(np.zeros(self.action_space.shape[0]))

        qpos = np.array(self._init_q)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)

        info = self.get_observation()
        obs_single, reward, terminated, truncated = self.task.reset(info, delta_action=np.zeros(self.action_space.shape[0]))
        self._obs_history = _NumpyObsRingBuffer(obs_single, self.history_len)
        ppo_frame, _ = self.compute_current_obs_for_the_ppo_policy(
            np.zeros(3, dtype=np.float32)
        )
        self._ppo_policy_obs_history = _NumpyObsRingBuffer(ppo_frame, self.history_len)
        observation = self._stacked_obs_from_buffer()
        self.last_step_time = time.time()

        return observation, info

    def get_joint_names(self):
        '''Returns the names of the joints.'''
        self.name_joints = []
        for i in range(1, self.model.njnt):  # skip root
            self.name_joints.append(mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        return self.name_joints

    def close(self):
        """Close rendering contexts processes."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    PATH_TO_PPO_AGENT = '../runs/26-05-02_22-13-21/checkpoints/best_agent.pt'
    # Base policy.
    from cleanrl_ppo import Agent
    import envs.biped as biped # TODO: Make this less stupid.
    env_jax = biped.VectorEnv(biped.BipedSim(), num_envs=1)
    agent_ppo = Agent(env_jax)

    import torch
    # Checkpoint stores a flat state_dict (actor_mean.*, critic.*, actor_logstd).
    ckpt = torch.load(PATH_TO_PPO_AGENT, map_location="cpu")
    agent_ppo.load_state_dict(ckpt["model"])

    counter = 0
    counter_max = 1000
    env = BipedSimpleEnv(render_mode="rgb_array",
                        control_dt=0.01,
                        base_policy=agent_ppo,
                        model_path='/data/robot_learning/robot_learning/src/assets/biped/xmls/biped_RL.xml')
    env = gym.wrappers.RecordVideo(env, "videos/biped_simple",
                                step_trigger=lambda x: x % counter_max == 0,
                                video_length=counter_max)
    try:
        while counter < 1000:
            print(f'---- step {counter} ----')
            env.step(np.array([0.0, 0.0]))

            time.sleep(0.001)
            counter += 1
    finally:
        env.close()

    print("Done")
