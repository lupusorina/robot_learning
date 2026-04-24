## Performance

<img src="./videos/video.gif" width="400"/>

## Run The Robot With The Keyboard

<img src="./videos/video_joystick.gif" width="400"/>

```bash
cd /home/adrian/robot_learning
```

```bash
conda activate /home/adrian/robot_learning/.conda/envs/robot-learning-gpu
```

```bash
cd robot_learning/src/jax/envs
```

```bash
mjpython biped_test.py
```

On macOS, you may need to allow input monitoring for both your terminal and
`mjpython`.

## Install

```bash
cd /home/adrian/robot_learning
```

```bash
conda env create -p ./.conda/envs/robot-learning-gpu -f environment_gpu.yml
```

```bash
conda activate /home/adrian/robot_learning/.conda/envs/robot-learning-gpu
```

```bash
python -m pip install --no-build-isolation -e .
```

If the Conda create step fails with an `x86_64-conda-linux-gnu-*` error:

```bash
conda deactivate
```

```bash
conda env create -p ./.conda/envs/robot-learning-gpu -f environment_gpu.yml
```

## Verify

```bash
python -c "import jax; print(jax.devices())"
```

You should see:

```bash
[CudaDevice(id=0)]
```

If you see CPU instead:

```bash
unset LD_LIBRARY_PATH
```

```bash
python -c "import jax; print(jax.devices())"
```

## Train

```bash
cd /home/adrian/robot_learning
```

```bash
conda activate /home/adrian/robot_learning/.conda/envs/robot-learning-gpu
```

```bash
python robot_learning/src/jax/train.py
```

Training results are written to:

```bash
results/<timestamp>/
```

To check progress:

```bash
ls -lt results | head
```

```bash
nvidia-smi
```
