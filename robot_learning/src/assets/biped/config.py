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

# Domain randomization.
RANDOMIZE_ARMATURE = True
ARMATURE_NOMINAL = 0.001
ARMATURE_MIN = 0.0001
ARMATURE_MAX = 0.005

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
N_SUBSTEPS = 10

LIMIT_OBSERVATIONS = 10.0

ADD_OBSERVATION_NOISE = True

OBS_NOISE_BASE_LIN_VEL = 0.3
OBS_NOISE_BASE_ANG_VEL = 0.3
OBS_NOISE_BASE_ROT = 0.05
OBS_NOISE_JOINT_POS = 0.01
OBS_NOISE_JOINT_VEL = 0.5