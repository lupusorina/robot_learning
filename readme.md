## Performance

<img src="./videos/video.gif" width="400"/>

# Run the robot with the keyboard

<img src="./videos/video_joystick.gif" width="400"/>

```
cd robot_learning/src/jax/envs
mjpython biped_test.py
```

Note: on MacOS you will likely need to whitelist both the terminal and the mjpython executable for input monitoring.
In System Settings, navigate to Privacy & Security -> Input Monitoring, then add your terminal and mjpython (cmd+shift+G in the file browser will let you add a path).

## GPU install

Tested on Linux with Python 3.12 and an NVIDIA GPU. The safest way to avoid
system-wide installs is to use the repo-local Conda environment defined in
`environment_gpu.yml`.

This is the currently tested GPU training stack:

```bash
brax==0.14.2
jax[cuda13]==0.9.2
mujoco==3.7.0
mujoco-mjx==3.7.0
```

### 1. Create the repo-local GPU environment

```bash
cd /home/adrian/robot_learning
conda env create -p ./.conda/envs/robot-learning-gpu -f environment_gpu.yml
conda activate /home/adrian/robot_learning/.conda/envs/robot-learning-gpu
```

If Conda fails during the pip phase with an error mentioning
`x86_64-conda-linux-gnu-*`, deactivate any previously active Conda env first and
rerun:

```bash
conda deactivate
conda env create -p ./.conda/envs/robot-learning-gpu -f environment_gpu.yml
conda activate /home/adrian/robot_learning/.conda/envs/robot-learning-gpu
```

### 2. Install the repo in editable mode

```bash
python -m pip install --no-build-isolation -e .
```

### 3. Verify GPU JAX

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Expected output on a working GPU setup:

```bash
0.9.2
[CudaDevice(id=0)]
```

If JAX falls back to CPU and you have `LD_LIBRARY_PATH` set, clear it for the
current shell and test again:

```bash
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.devices())"
```

### 4. Launch training

```bash
cd /home/adrian/robot_learning
conda activate /home/adrian/robot_learning/.conda/envs/robot-learning-gpu
python robot_learning/src/jax/train.py
```

Successful startup should include:

```bash
Available devices: [CudaDevice(id=0)]
```

As training progresses, you should start seeing lines like:

```bash
Reward for 14417920 steps: 26.773
Reward for 28835840 steps: 33.218
```

Training artifacts are written under:

```bash
results/<timestamp>/
```

You should see files such as:

```bash
reward.pdf
reward.png
config.json
biped_RL.xml
```

### 5. Monitor the run

Check that a new results directory is being updated:

```bash
ls -lt results | head
ls -lt results/<latest-timestamp>
```

Check GPU activity:

```bash
nvidia-smi
```

## Legacy environment

`environment.yml` is kept as the older compatibility environment. It is not the
recommended path for modern NVIDIA GPU training on newer hardware.

Inference: `test.ipynb`



## File structure

```
robot_learning
    └── src
            └── jax
                └── biped.py                   (Biped in Jax)
                └── train.py                   (Train PPO on Biped)
                └── test.ipynb                 (Jupyter notebook for testing)
                └── mjx_env.py                 (file taken from mujoco-playground and modified)
                └── wrapper.py                 (file taken from mujoco-playground and modified)
                └── randomize.py               (domain randomization)
                └── utils.py                   (utils)
            └── assets
                └── biped                      (biped)

    └── tests
```


This project uses/derives from MuJoCo Playground (Apache 2.0) by Google DeepMind.
