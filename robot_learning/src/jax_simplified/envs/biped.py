import os
if 'DISPLAY' not in os.environ:
    # enable headless rendering with EGL if no display is available
    os.environ['MUJOCO_GL'] = 'egl'

# by default XLA pre-allocates much more GPU memory than needed, this reduces it
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

import jax
import time
import mujoco
import mujoco.mjx
import numpy as np
from flax import struct
import gymnasium as gym
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from mujoco.mjx._src import math as mjx_math
import robot_learning.src.assets.biped.config as robot_config

# Custom imports.
import utils
from utils import MjxRenderer, save_video, tile_images
from utils import RingBuffer

if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Notes:
# - MuJoCo quaternion order is [w, x, y, z]
# - MuJoCo qpos order for 6dof is [position, quaternion]
# - joint omega is in the body frame, joint velocity is in the world frame
#    (see https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst#floating-objects)


def replace_pytree(pytree, replacements: dict[str, jax.Array]):
    def update_fn(path, value):
        return replacements.get(jax.tree_util.keystr(path), value)
    return jax.tree_util.tree_map_with_path(update_fn, pytree)


class BipedSim:
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,))

    @struct.dataclass
    class BipedState:
        mjdata: mujoco.mjx.Data
        command: jax.Array
        last_act: jax.Array
        last_last_act: jax.Array
        phase: jax.Array
        phase_dt: jax.Array
        feet_air_time: jax.Array
        last_contact: jax.Array
        swing_peak: jax.Array
        step_count: jax.Array
        reward: jax.Array
        done: jax.Array
        obs_history: RingBuffer

    @struct.dataclass
    class BipedModel:
        mjmodel: mujoco.mjx.Model

    def __init__(self):

        self._mjx_math = mjx_math
        self._robot_config = robot_config
        self._model = mujoco.MjModel.from_xml_path(robot_config.XML_PATH)
        self._model.opt.timestep = 0.001
        self.ctrl_dt = 0.01
        self._sim_dt = 0.001
        self._n_substeps = int(round(self.ctrl_dt / self._sim_dt))
        self.dt = self.ctrl_dt
        self.history_len = 3

        self._mjx_model = mujoco.mjx.put_model(self._model)
        self._init_q = jnp.array(self._model.keyframe("home").qpos)
        self._default_q_joints = jnp.array(self._model.keyframe("home").qpos[7:])
        self._base_height_target = float(robot_config.DESIRED_HEIGHT)
        self._max_foot_height = 0.15
        self._soft_joint_pos_limit_factor = 0.95
        self._reward_scales = {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": 0.0,
            "ang_vel_xy": -0.15,
            "orientation": -1.0,
            "base_height": 0.0,
            "torques": -2.5e-4,
            "action_rate": -2e-4,
            "feet_air_time": 2.0,
            "feet_slip": -0.25,
            "feet_height": 0.0,
            "feet_phase": 1.0,
            "alive": 0.0,
            "termination": -1.0,
            "joint_deviation_knee": -0.1,
            "joint_deviation_hip": -0.25,
            "dof_pos_limits": -1.0,
            "pose": -1.0,
        }

        self._q_j_min, self._q_j_max = self._model.jnt_range[1:].T
        c = (self._q_j_min + self._q_j_max) / 2
        r = self._q_j_max - self._q_j_min
        self._soft_q_j_min = c - 0.5 * r * self._soft_joint_pos_limit_factor
        self._soft_q_j_max = c + 0.5 * r * self._soft_joint_pos_limit_factor

        self._feet_site_id = np.array([self._model.site(name).id for name in robot_config.FEET_SITES])
        self._feet_geom_id = np.array([self._model.geom(name).id for name in robot_config.FEET_GEOMS])
        self._floor_geom_id = self._model.geom("floor").id

        def get_joint_indices(joint_names, sides=robot_config.SIDES):
            return jnp.array([
                self._model.joint(f"{side}_{joint_name}").qposadr[0] - 7
                for side in sides
                for joint_name in joint_names
            ])

        self._hip_indices = get_joint_indices(robot_config.HIP_JOINT_NAMES)
        self._knee_indices = get_joint_indices(robot_config.KNEE_JOINT_NAMES)

        # Mapping from joint names to the PPO action indices.
        # Initialize the action space.
        self.idx_actuators_dict = {}
        for i in range(0, self._model.nu):
            self.idx_actuators_dict[mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] = i

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
        self.joint_names_to_actuator_idx_dict = { name: int(self._model.joint(name).qposadr[0] - 7) \
                                                for name in self.idx_actuators_dict.keys() }
        self.policy_idx_to_mujoco_actuator_idx_dict = { self.actuated_joint_names_to_policy_idx_dict[name]: self.joint_names_to_actuator_idx_dict[name] 
                                    for name in self.actuated_joint_names_to_policy_idx_dict }

        self._sensor_adr = {}
        for name in [
            robot_config.GRAVITY_SENSOR,
            robot_config.LOCAL_LINVEL_SENSOR,
            robot_config.GYRO_SENSOR,
            robot_config.GLOBAL_LINVEL_SENSOR,
            robot_config.GLOBAL_ANGVEL_SENSOR,
        ]:
            sid = self._model.sensor(name).id
            adr = self._model.sensor_adr[sid]
            dim = self._model.sensor_dim[sid]
            self._sensor_adr[name] = (adr, dim)

        foot_global_linvel_sensor_adr = []
        for site in robot_config.FEET_SITES:
            sensor_id = self._model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._model.sensor_adr[sensor_id]
            sensor_dim = self._model.sensor_dim[sensor_id]
            foot_global_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_global_linvel_sensor_adr = jnp.array(foot_global_linvel_sensor_adr)

        obs, _ = self.build_obs(self.reset(self.make_model(), jax.random.PRNGKey(0)), jax.random.PRNGKey(0))
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=obs.shape)

    def _sensor_data(self, data: mujoco.mjx.Data, sensor_name: str) -> jax.Array:
        adr, dim = self._sensor_adr[sensor_name]
        return data.sensordata[adr:adr + dim]

    def _geoms_colliding(self, data: mujoco.mjx.Data, geom1: int, geom2: int) -> jax.Array:
        mask = (jnp.array([geom1, geom2]) == data.contact.geom).all(axis=1)
        mask |= (jnp.array([geom2, geom1]) == data.contact.geom).all(axis=1)
        idx = jnp.where(mask, data.contact.dist, 1e4).argmin()
        dist = data.contact.dist[idx] * mask[idx]
        return dist < 0

    def get_randomized_model_params(self, rng: jax.Array) -> dict[str, jax.Array]:
        del rng
        return {}

    def make_model(self, randomized_params: dict[str, jax.Array] = {}) -> BipedModel:
        del randomized_params
        return self.BipedModel(mjmodel=self._mjx_model)

    def get_mjmodel(self, randomized_params: dict[str, jax.Array] = {}) -> mujoco.MjModel:
        del randomized_params
        return self._model

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(rng1, (), minval=-0.2, maxval=0.2)
        lin_vel_y = jax.random.uniform(rng2, (), minval=-0.2, maxval=0.2)
        ang_vel_yaw = jax.random.uniform(rng3, (), minval=-0.2, maxval=0.2)

        # With 10% chance, set everything to zero.
        return jnp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jnp.zeros(3),
            jnp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )
        
    def reset(self, model: BipedModel, rng: jax.Array) -> BipedState:
        qpos = self._init_q
        qvel = jnp.zeros(model.mjmodel.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = self._mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        qpos = qpos.at[3:7].set(self._mjx_math.quat_mul(qpos[3:7], quat))

        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(qpos[7:] * (1.0 + jax.random.uniform(key, (self._model.nu,), minval=-0.1, maxval=0.1)))
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-1.0, maxval=1.0))

        data = mujoco.mjx.make_data(model.mjmodel).replace(qpos=qpos, qvel=qvel, ctrl=qpos[7:])
        data = mujoco.mjx.forward(model.mjmodel, data)

        rng, key = jax.random.split(rng)
        command = self._sample_command(key)
        phase = jnp.array([0.0, jnp.pi])
        phase_dt = 2 * jnp.pi * self.ctrl_dt * jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        current_obs = self._compute_current_obs(data, command, jnp.zeros(self.action_space.shape[0]), phase)
        obs_hist = RingBuffer.init(current_obs, self.history_len)

        return self.BipedState(
            mjdata=data,
            command=command,
            last_act=jnp.zeros(self.action_space.shape[0]),
            last_last_act=jnp.zeros(self.action_space.shape[0]),
            phase=phase,
            phase_dt=phase_dt,
            feet_air_time=jnp.zeros(2),
            last_contact=jnp.zeros(2, dtype=bool),
            swing_peak=jnp.zeros(2),
            step_count=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(0.0),
            done=jnp.array(False),
            obs_history=obs_hist,
        )

    def _compute_current_obs(self, data: mujoco.mjx.Data, command: jax.Array, last_act: jax.Array, phase: jax.Array) -> jax.Array:
        linvel = self._sensor_data(data, self._robot_config.LOCAL_LINVEL_SENSOR)
        gyro = self._sensor_data(data, self._robot_config.GYRO_SENSOR)
        up_B = self._sensor_data(data, self._robot_config.GRAVITY_SENSOR)
        q_joints = data.qpos[7:]
        q_vel = data.qvel[6:]
        ph = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])
        return jnp.concatenate([
            linvel,
            gyro,
            up_B,
            command,
            q_joints - self._default_q_joints,
            q_vel,
            last_act,
            ph,
        ]).clip(-10.0, 10.0)

    def step(self, model: BipedModel, state: BipedState, action: jax.Array) -> BipedState:
         # Step the model.
        action_complete = jnp.zeros(self._model.nu)
        for _, policy_idx in self.actuated_joint_names_to_policy_idx_dict.items():
            if policy_idx is None:
                continue
        action_complete = action_complete.at[self.policy_idx_to_mujoco_actuator_idx_dict[policy_idx]].set(action[policy_idx])

        motor_targets = self._default_q_joints + action_complete

        data = state.mjdata

        def sim_step(d, _):
            d = d.replace(ctrl=motor_targets)
            d = mujoco.mjx.step(model.mjmodel, d)
            return d, None

        data, _ = jax.lax.scan(sim_step, data, (), self._n_substeps)
        contact = jnp.array([self._geoms_colliding(data, g, self._floor_geom_id) for g in self._feet_geom_id])
        first_contact = (state.feet_air_time > 0.0) * (contact | state.last_contact)
        feet_pos = data.site_xpos[self._feet_site_id]
        swing_peak = jnp.maximum(state.swing_peak, feet_pos[..., -1])

        done = self._termination(data)
        reward = self._reward(data, action, state, first_contact, contact, done)

        phase = jnp.fmod(state.phase + state.phase_dt + jnp.pi, 2 * jnp.pi) - jnp.pi
        current_obs = self._compute_current_obs(data, state.command, action, phase)
        obs_hist = RingBuffer.push(state.obs_history, current_obs)

        return state.replace(
            mjdata=data,
            last_last_act=state.last_act,
            last_act=action,
            phase=phase,
            feet_air_time=jnp.where(contact, 0.0, state.feet_air_time + self.ctrl_dt),
            last_contact=contact,
            swing_peak=jnp.where(contact, 0.0, swing_peak),
            step_count=jnp.where(done | (state.step_count > 500), 0, state.step_count + 1),
            reward=reward,
            done=done,
            obs_history=obs_hist,
        )

    def _termination(self, data: mujoco.mjx.Data) -> jax.Array:
        gravity = self._sensor_data(data, self._robot_config.GRAVITY_SENSOR)
        return (gravity[-1] < 0.0) | jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()

    def _reward(
        self,
        data: mujoco.mjx.Data,
        action: jax.Array,
        state: BipedState,
        first_contact: jax.Array,
        contact: jax.Array,
        done: jax.Array,
    ) -> jax.Array:
        cmd = state.command
        lin = self._sensor_data(data, self._robot_config.LOCAL_LINVEL_SENSOR)
        gyro = self._sensor_data(data, self._robot_config.GYRO_SENSOR)
        global_linvel = self._sensor_data(data, self._robot_config.GLOBAL_LINVEL_SENSOR)
        global_angular_vel = self._sensor_data(data, self._robot_config.GLOBAL_ANGVEL_SENSOR)
        grav = self._sensor_data(data, self._robot_config.GRAVITY_SENSOR)

        # Tracking rewards.
        tracking_lin = jnp.exp(-jnp.sum(jnp.square(cmd[:2] - lin[:2]))) # OK.
        tracking_ang = jnp.exp(-jnp.square(cmd[2] - gyro[2])) # OK.
        lin_vel_z = jnp.square(global_linvel[2])
        ang_vel_xy = jnp.sum(jnp.square(global_angular_vel[:2])) # OK.
        torso_zaxis = jnp.sum(jnp.square(grav[:2])) # OK.
        base_height = jnp.square(data.qpos[2] - self._base_height_target)
        
        # Energy related rewards.
        torques = jnp.sum(jnp.abs(data.actuator_force)) # OK.
        action_rate = jnp.sum(jnp.square(action - state.last_act) / self.ctrl_dt) # OK.

        # Feet related rewards.
        feet_vel_xy = data.sensordata[self._foot_global_linvel_sensor_adr][..., :2]
        feet_slip = jnp.sum(jnp.linalg.norm(feet_vel_xy, axis=-1) * contact)

        cmd_norm = jnp.linalg.norm(cmd)
        threshold_min = 0.2; 
        threshold_max = 0.5
        air_time = (state.feet_air_time - threshold_min) * first_contact
        air_time = jnp.clip(air_time, max=threshold_max - threshold_min)
        feet_air_time = jnp.sum(air_time)
        feet_air_time *= cmd_norm > 0.1  # No reward for zero commands.

        feet_height_error = state.swing_peak / self._max_foot_height - 1.0
        feet_height = jnp.sum(jnp.square(feet_height_error) * first_contact)

        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        rz = utils.get_rz(state.phase, swing_height=self._max_foot_height)
        error = jnp.sum(jnp.square(foot_z - rz))
        feet_phase = jnp.exp(-error / 0.01)

        joint_deviation_hip = jnp.sum(
            jnp.abs(data.qpos[7:][self._hip_indices] - self._default_q_joints[self._hip_indices])
        ) * (jnp.abs(cmd[1]) > 0.1)
        joint_deviation_knee = jnp.sum(
            jnp.abs(data.qpos[7:][self._knee_indices] - self._default_q_joints[self._knee_indices])
        )
        out_of_limits = -jnp.clip(data.qpos[7:] - self._soft_q_j_min, None, 0.0)
        out_of_limits += jnp.clip(data.qpos[7:] - self._soft_q_j_max, 0.0, None)
        dof_pos_limits = jnp.sum(out_of_limits)

        # Keep the robot close to the default pose.
        pose = jnp.sum(jnp.square(data.qpos[7:] - self._default_q_joints))

        reward_terms = {
            "tracking_lin_vel": tracking_lin,
            "tracking_ang_vel": tracking_ang,
            "lin_vel_z": lin_vel_z,
            "ang_vel_xy": ang_vel_xy,
            "orientation": torso_zaxis,
            "base_height": base_height,
            "torques": torques,
            "action_rate": action_rate,
            "feet_air_time": feet_air_time,
            "feet_slip": feet_slip,
            "feet_height": feet_height,
            "feet_phase": feet_phase,
            "alive": jnp.array(1.0),
            "termination": done,
            "joint_deviation_knee": joint_deviation_knee,
            "joint_deviation_hip": joint_deviation_hip,
            "dof_pos_limits": dof_pos_limits,
            "pose": pose,
        }
        total = sum(self._reward_scales[k] * v for k, v in reward_terms.items())
        return jnp.clip(total * self.ctrl_dt, 0.0, 10000.0)

    def build_obs(self, state: BipedState, rng: jax.Array) -> tuple[jax.Array, dict]:
        del rng
        hist = RingBuffer.get(state.obs_history, jnp.arange(self.history_len)).reshape(-1)
        obs = hist[::-1] # Most recent observation first.
        info = {
            "state": state,
            "obs": obs,
            "benchmark_reward": state.reward,
        }
        return obs, info

    def get_reward(self, info: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
        s = info["state"]
        return s.reward, s.done, jnp.array(False)

    def get_renderer(self, randomized_params: dict[str, jax.Array] = {}) -> MjxRenderer:
        del randomized_params
        return MjxRenderer(self._model)

    @staticmethod
    def extract_render_state(state: "BipedSim.BipedState") -> jax.Array:
        return state.mjdata.qpos


class VectorEnv(gym.vector.VectorEnv):
    """A vectorized, JIT-compiled batched environment."""
    def __init__(self, sim, num_envs=1, seed=0, backend=None):
        self.render_mode = "rgb_array"
        self.dt = sim.dt
        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}
        self.single_observation_space = sim.observation_space
        self.single_action_space = sim.action_space
        self.rng = jax.random.PRNGKey(seed)
        self.sim = sim
        self.num_envs = num_envs
        self.model = self.sim.make_model()
        randomized_keys = set(self.sim.get_randomized_model_params(jax.random.PRNGKey(0)).keys())
        existing_keys = set([jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_leaves_with_path(self.model)])
        missing_keys = randomized_keys - existing_keys
        if missing_keys:
            print(existing_keys)
            raise RuntimeError(f"Randomized keys {missing_keys} not found in model")
        def batch_axes_randomized(path, _):
            return 0 if jax.tree_util.keystr(path) in randomized_keys else None
        self.model_batch_axes = jax.tree_util.tree_map_with_path(batch_axes_randomized, self.model)
        self._define_jit_functions(backend)
        self.randomize_model()
        self.reset()

    def _define_jit_functions(self, backend):
        def vmapped_reset(model, rng):
            print("tracing reset")
            rng = jax.random.split(rng, self.num_envs)
            return jax.vmap(self.sim.reset, in_axes=[self.model_batch_axes, 0])(model, rng)
        def vmapped_autoreset(reset_mask, states, reset_states):
            def reset_if_done(x, y):
                return jnp.where(jnp.reshape(reset_mask, [reset_mask.shape[0]] + [1]*(len(x.shape) -1)), y, x)
            return jax.tree_util.tree_map(reset_if_done, states, reset_states)
        def vmapped_step(model, states, actions):
            return jax.vmap(self.sim.step, in_axes=[self.model_batch_axes, 0, 0])(model, states, actions)
        def vmapped_build_obs(state, rng):
            rng = jax.random.split(rng, self.num_envs)
            return jax.vmap(self.sim.build_obs)(state, rng)
        def vmapped_get_reward(info):
            return jax.vmap(self.sim.get_reward)(info)
        def vmapped_randomize_model(model, rng):
            keys = jax.random.split(rng, self.num_envs)
            randomized_params = jax.vmap(self.sim.get_randomized_model_params)(keys)
            return replace_pytree(model, randomized_params), randomized_params
        def vmapped_get_render_state(states):
            return jax.vmap(self.sim.extract_render_state, in_axes=[0])(states)
        self._vmapped_reset = jax.jit(vmapped_reset, backend=backend)
        self._vmapped_autoreset = jax.jit(vmapped_autoreset, backend=backend)
        self._vmapped_step = jax.jit(vmapped_step, backend=backend)
        self._vmapped_build_obs = jax.jit(vmapped_build_obs, backend=backend)
        self._vmapped_get_reward = jax.jit(vmapped_get_reward, backend=backend)
        self._vmapped_randomize_model = jax.jit(vmapped_randomize_model, backend=backend)
        self._vmapped_get_render_state = jax.jit(vmapped_get_render_state, backend=backend)

    def reset(self, seed: int = None, options: dict = {}) -> tuple[jax.Array, dict]:
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.randomize_model()
        self.rng, key = jax.random.split(self.rng)
        self.states = self._vmapped_reset(self.model, key)
        self.reset_states = self.states
        self.reset_next = jnp.zeros(self.num_envs, dtype=jnp.bool_)
        self.rng, key = jax.random.split(self.rng)
        obs, info = self._vmapped_build_obs(self.states, key)
        return np.array(obs), info

    def randomize_model(self):
        self.rng, key = jax.random.split(self.rng)
        self.model, self.randomized_params = self._vmapped_randomize_model(self.model, key)
        self.renderers = [] # clear renderers

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        self.states = self._vmapped_step(self.model, self.states, actions)
        self.states = self._vmapped_autoreset(self.reset_next, self.states, self.reset_states)
        self.rng, key = jax.random.split(self.rng)
        obs, info = self._vmapped_build_obs(self.states, key)
        reward, terminated, truncated = self._vmapped_get_reward(info)
        self.reset_next = terminated | truncated
        return np.array(obs), np.array(reward), np.array(terminated), np.array(truncated), info

    def render(self, max_render_envs=16):
        nb_render_envs = int(min(max_render_envs, self.num_envs))
        if len(self.renderers) < nb_render_envs:
            for i in range(len(self.renderers), nb_render_envs):
                print(f"creating renderer for env {i}")
                renderer = self.sim.get_renderer(
                    jax.tree_util.tree_map(lambda x: x[i], self.randomized_params)
                )
                self.renderers.append(renderer)
        images = []
        render_states = self._vmapped_get_render_state(self.states)
        for i in range(nb_render_envs):
            # images.append(self.renderers[i].render(jax.tree_util.tree_map(lambda x: x[i], self.states)))
            images.append(self.renderers[i].render(render_states[i]))
        return images



if __name__ == "__main__":
    print("building sim")
    sim = BipedSim()

    start_build_time = time.time()
    env = VectorEnv(sim, num_envs=1024)
    obs, info = env.reset()
    # print(obs)
    # print(info)

    obs, reward, terminated, truncated, info = env.step(jnp.zeros([env.num_envs, sim.action_space.shape[0]]))
    print("step done")
    obs, reward, terminated, truncated, info = env.step(jnp.zeros([env.num_envs, sim.action_space.shape[0]]))
    print("step done")

    print(f"build time: {time.time() - start_build_time}")

    start_time = time.time()

    nb_envs_to_render = 6
    frames = []
    total_render_time = 0
    steps = 200
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(jnp.zeros([env.num_envs, sim.action_space.shape[0]]))
        start_render_time = time.time()    
        frames.append(tile_images(env.render(nb_envs_to_render)))
        total_render_time += time.time() - start_render_time
        print(f"benchmark_reward: {info['benchmark_reward'][0]}")
    print(f"render time: {total_render_time}")
    print(f"total time taken: {time.time() - start_time}")
    print(f"steps per second: {steps * env.num_envs / (time.time() - start_time)}")

    save_video(frames, "batch_render.mp4", fps=1/env.dt)
    print(f"final time: {info['state'].mjdata.time[0]}")


    stats = jax.devices()[0].memory_stats()
    print(f"peak memory usage: {stats['peak_bytes_in_use'] / 1024 / 1024 / 1024:.2f} GB")
