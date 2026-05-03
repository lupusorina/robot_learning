import os
import time

import numpy as np
import wandb


def _to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(np.array(value).mean())


def _format_hhmmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingLogger:
    def __init__(
        self,
        *,
        use_wandb: bool,
        run_name: str,
        config: dict,
        video_enabled: bool,
        video_interval_iters: int,
        video_duration_iters: int,
        video_dir: str,
        video_fps: float,
        video_frame_stride: int,
    ):
        if video_enabled and (video_duration_iters <= 0 or video_interval_iters <= 0):
            raise ValueError("VIDEO_DURATION_ITERS and VIDEO_INTERVAL_ITERS must be positive integers")
        if video_enabled and video_frame_stride <= 0:
            raise ValueError("video_frame_stride must be a positive integer")

        self.use_wandb = use_wandb
        self.video_enabled = video_enabled
        self.video_interval_iters = video_interval_iters
        self.video_duration_iters = video_duration_iters
        self.video_dir = video_dir
        self.video_fps = video_fps
        self.video_frame_stride = video_frame_stride

        self.train_wall_start_s = time.time()
        self.iter_wall_start_s = self.train_wall_start_s
        self.video_frames = []
        self.video_frame_count = 0
        self.video_recording_active = False
        self.video_start_iter = 0
        self.video_end_iter = 0
        self.rollout_done_fracs = []
        self.rollout_reward_terms_means = {}

        if self.use_wandb:
            wandb.init(project="robot_learning", name=run_name, config=config)

    def start_iteration(self, iteration: int, nb_iterations: int) -> None:
        self.iter_wall_start_s = time.time()
        self.rollout_done_fracs = []
        self.rollout_reward_terms_means = {}

        if self.video_enabled and (not self.video_recording_active) and (
            iteration == 1 or (iteration % self.video_interval_iters == 0)
        ):
            self.video_recording_active = True
            self.video_start_iter = iteration
            self.video_end_iter = min(nb_iterations, iteration + self.video_duration_iters - 1)
            self.video_frames = []
            self.video_frame_count = 0

    def record_step(self, infos: dict, terminated, truncated, env_not_wrapped) -> None:
        if (not self.use_wandb) and (not self.video_recording_active):
            return

        if self.video_recording_active:
            if self.video_frame_count % self.video_frame_stride == 0:
                from utils import tile_images

                self.video_frames.append(tile_images(env_not_wrapped.render()))
            self.video_frame_count += 1

        if not self.use_wandb:
            return

        dones = terminated | truncated
        self.rollout_done_fracs.append(_to_float(dones))

        if "reward_terms" in infos:
            for key, value in infos["reward_terms"].items():
                self.rollout_reward_terms_means.setdefault(key, []).append(_to_float(value))

    def end_iteration(
        self,
        *,
        iteration: int,
        nb_iterations: int,
        timestep: int,
        num_envs: int,
        average_reward,
    ) -> None:
        iter_wall_s = time.time() - self.iter_wall_start_s
        train_elapsed_s = time.time() - self.train_wall_start_s
        avg_iter_wall_s = train_elapsed_s / max(iteration, 1)
        eta_s = (nb_iterations - iteration) * avg_iter_wall_s
        average_reward_float = _to_float(average_reward)

        print(
            f"Iteration {iteration}/{nb_iterations} "
            f"reward={average_reward_float:.5f} "
            f"iter_s={iter_wall_s:.2f} "
            f"elapsed={_format_hhmmss(train_elapsed_s)} "
            f"eta={_format_hhmmss(eta_s)}"
        )

        if self.video_recording_active and iteration >= self.video_end_iter:
            if self.use_wandb and self.video_frames:
                video_iteration_rounded = int(((int(self.video_start_iter) + 25) // 50) * 50)
                frames = np.stack(self.video_frames).astype(np.uint8).transpose(0, 3, 1, 2)
                wandb.log(
                    {
                        "video": wandb.Video(frames, fps=int(round(self.video_fps)), format="gif"),
                        "video_iteration_raw": int(self.video_start_iter),
                        "video_iteration": video_iteration_rounded,
                    },
                    step=int(timestep * num_envs),
                )
            self.video_recording_active = False
            self.video_frames = []

        if self.use_wandb:
            done_frac_mean = 0.0
            if self.rollout_done_fracs:
                done_frac_mean = float(np.mean(self.rollout_done_fracs))

            reward_terms_payload = {}
            for key, values in self.rollout_reward_terms_means.items():
                reward_terms_payload[f"reward_terms/{key}"] = float(np.mean(values))

            wandb.log(
                {
                    "iteration": int(iteration),
                    "iteration/total": int(nb_iterations),
                    "reward/mean": average_reward_float,
                    "termination/done_frac_mean": done_frac_mean,
                    **reward_terms_payload,
                },
                step=int(timestep * num_envs),
            )
