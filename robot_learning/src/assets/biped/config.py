import os

parent_dir = os.path.abspath(os.path.join(os.getcwd()))
XML_PATH = os.path.join(parent_dir, '../assets/biped/xmls/biped_point_feet_RL.xml')
# XML_PATH = os.path.join(parent_dir, 'robot_learning/src/assets/biped/xmls/biped_point_feet_RL.xml')

#XML_PATH = os.path.join(parent_dir, '../../assets/biped/xmls/biped_point_feet_RL.xml')

ROOT_BODY = "base_link"
FEET_SITES = ["l_foot", "r_foot"]
LEFT_FEET_GEOMS = ["L_FOOT"]
RIGHT_FEET_GEOMS = ["R_FOOT"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
IMU_SITE = "imu_location"
DESIRED_HEIGHT = 0.56
DESIRED_FOOT_HEIGHT = 0.15
FEET_PHASE_STD = 0.06
GAIT_PERIOD = 0.70
COMMAND_RESAMPLE_INTERVAL = 500

# Domain randomization.
PUSH_ENABLE = True
PUSH_INTERVAL_RANGE = (5.0, 10.0)
PUSH_MAGNITUDE_RANGE = (0.05, 1.0)

RANDOMIZE_ARMATURE = True
ARMATURE_NOMINAL = 0.001
ARMATURE_MIN = 0.0001
ARMATURE_MAX = 0.005

# Model parameter randomization (mirrors robot_learning/src/jax/randomize.py).
RANDOMIZE_FLOOR_FRICTION = False
FLOOR_FRICTION_RANGE = (0.4, 1.0)

RANDOMIZE_LINK_MASSES = False
LINK_MASS_RANGE = (0.9, 1.1)

RANDOMIZE_TORSO_MASS = False
TORSO_MASS_RANGE = (-0.5, 0.5)

RANDOMIZE_QPOS0 = False
QPOS0_JITTER_RANGE = (-0.1, 0.1)

RANDOMIZE_BODY_IPOS = False
BODY_IPOS_MIN = (-0.05, -0.02, -0.005)
BODY_IPOS_MAX = (0.05, 0.02, 0.005)

RANDOMIZE_ACTUATOR_GAINS = False
ACTUATOR_GAIN_RANGE = (0.9, 1.1)

# Normalized actions in [-1, 1] map to these joint-position offsets around the
# XML home pose. Each tuple is (negative_action_offset, positive_action_offset).
ACTION_TARGET_OFFSETS = {
    "L_HAA": (-0.20, 0.20),
    "L_HFE": (-0.70, 0.90),
    "L_KFE": (-0.90, 1.10),
    "R_HAA": (-0.20, 0.20),
    "R_HFE": (-0.70, 0.90),
    "R_KFE": (-0.90, 1.10),
}

HIP_JOINT_NAMES = ["HAA", "HFE"]
KNEE_JOINT_NAMES = ["KFE"]
# ANKLE_FE_JOINT_NAMES = ["ANKLE"]
ANKLE_FE_JOINT_NAMES = []
ANKLE_AA_JOINT_NAMES = []

SIDES = ["L", "R"]

                    # L_HAA L_HFE L_KFE
                    # R_HAA R_HFE R_KFE
COSTS_JOINT_ANGLES = [1.0, 0.01, 0.01, 1.0, 0.01, 0.01]


CTRL_DT = 0.01
SIM_DT = 0.002
HISTORY_LEN = 3

LIMIT_OBSERVATIONS = 10.0

ADD_OBSERVATION_NOISE = True

OBS_NOISE_BASE_LIN_VEL = 0.3
OBS_NOISE_BASE_ANG_VEL = 0.3
OBS_NOISE_BASE_ROT = 0.05
OBS_NOISE_JOINT_POS = 0.01
OBS_NOISE_JOINT_VEL = 0.5

# Observation scaling: each block is multiplied by its factor (elementwise on that
# block). Applied after additive observation noise for the actor, then clipped.
# All keys live in OBS_SCALE. Keys used only on the critic tail are marked in the
# comments below; the rest appear on the actor and are reused on the critic where
# the same quantity is repeated (gyro, upvector, linvel, joint errors, joint vels).
OBS_SCALE = {
    "base_lin_vel": 1.0,
    "base_ang_vel": 0.25,
    "upvector": 1.0,
    "command": 1.0,
    "joint_pos_err": 0.5,
    "joint_vel": 0.05,
    "last_action": 1.0,
    "phase": 1.0,
    # Critic suffix only (after policy-shaped prefix):
    "accelerometer": 0.05,
    "global_ang_vel": 1.0,
    "baselink_height": 1.0,
    "actuator_force": 0.5,
    "contact": 1.0,
    "feet_lin_vel": 0.25,
    "feet_air_time": 1.0,
}
