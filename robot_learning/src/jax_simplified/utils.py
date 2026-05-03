# Modified from MuJoCo Playground (Apache 2.0) by Google DeepMind.

import os
import jax
import math
import mujoco
import imageio
import numpy as np
from mujoco import mjx
import jax.numpy as jnp
import gymnasium as gym
from flax import struct
from typing import Union
from typing import Any, Tuple, Union

def get_rz(
    phi: Union[jax.Array, float], 
    swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jnp.pi) / (2 * jnp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jnp.where(x <= 0.5, stance, swing)


def get_foot_pos_z(
    phase: Union[jax.Array, float],
    swing_height: Union[jax.Array, float] = 0.10,
) -> jax.Array:
    swing_z = swing_height * jnp.sin(phase)
    return jnp.where(phase > 0.0, swing_z, 0.0)


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jnp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """Return True if the two geoms are colliding."""
  return get_collision_info(state.contact, geom1, geom2)[0] < 0


## Ring buffer.

@struct.dataclass
class RingBuffer:
    buf: Any        # pytree, leaves shaped (N, ...)
    idx: jax.Array  # scalar int32
    N: int  # number of elements in the buffer

    @staticmethod
    def init(example, N: int) -> "RingBuffer":
        buf = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (N,) + x.shape),
            example,
        )
        return RingBuffer(buf=buf, idx=jnp.int32(0), N=N)

    @staticmethod
    def push(rb: "RingBuffer", x) -> "RingBuffer":
        buf = jax.tree.map(
            lambda b, xi: b.at[rb.idx].set(xi),
            rb.buf,
            x,
        )
        idx = (rb.idx + 1) % rb.N
        return RingBuffer(buf=buf, idx=idx, N=rb.N)

    @staticmethod
    def get(rb: "RingBuffer", k: jax.Array):
        """k=0 -> most recent"""
        i = (rb.idx - 1 - k) % rb.N
        return jax.tree.map(lambda b: b[i], rb.buf)


## Rendering.
class SaveVideoWrapper(gym.vector.vector_env.VectorWrapper):
    def __init__(self, env, video_path, nb_envs_to_render=4, nb_steps_per_video=400):
        super().__init__(env)
        self.video_path = video_path
        self.frames = []
        self.nb_envs_to_render = nb_envs_to_render
        self.counter = 0
        self.nb_steps_per_video = nb_steps_per_video
        self.step_counter = 0

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.step_counter += 1
        if self.step_counter <= self.nb_steps_per_video:
            self.frames.append(tile_images(self.env.render(self.nb_envs_to_render)))
        elif self.frames:
            save_video(self.frames, self.video_path + f"/video_{self.counter:04d}.mp4", fps=1/self.env.dt)
            self.counter += 1
            self.frames = []
        else:
            pass
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = {}) -> tuple[jax.Array, dict]:
        obs, info = self.env.reset(seed, options)
        self.step_counter = 0
        self.frames = []
        return obs, info
      
class MjxRenderer:
    def __init__(self, model: mujoco.MjModel, height=240, width=320):
        self.model = model
        self.mj_data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, qpos: jax.Array):
        self.mj_data.qpos[:] = np.array(qpos)
        mujoco.mj_forward(self.model, self.mj_data)
        self.renderer.update_scene(self.mj_data, camera=-1)
        return self.renderer.render()
      
  
def save_video(images, filename, fps=30):
    images = [np.array(img) for img in images]
    images = np.array(images)
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    imageio.mimwrite(filename, images, fps=fps)

def tile_images(images: tuple[np.ndarray]) -> np.ndarray:
    """Tile a list of images into a nearly square grid (close to square layout)."""
    images = list(images)
    n = len(images)
    # Compute grid size (rows, cols) to make as close to square as possible
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # Get image height and width
    h, w = images[0].shape[:2]
    # Fill extra spots with black images if needed
    images_padded = images + [np.zeros_like(images[0]) for _ in range(rows * cols - n)]
    # Stack images row by row
    img_rows = []
    for i in range(rows):
        img_row = np.concatenate(images_padded[i * cols:(i + 1) * cols], axis=1)
        img_rows.append(img_row)
    grid = np.concatenate(img_rows, axis=0)
    return grid


## Github.
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
