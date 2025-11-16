## Performance

<img src="./videos/video.gif" width="400"/>

# Run the robot with the keyboard

<img src="./videos/video_joystick.gif" width="400"/>

```
cd robot_learning/src/jax/envs
mjpython biped_test.py
```

## Install requirements

Tested on Python 3.12.

```
pip3 install -r requirements.txt
```

## Install the repo

```
pip3 install -e .
```

## Run biped learning code in Jax

```
cd src/jax
python3 train.py
```
Inference: test.ipynb



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
