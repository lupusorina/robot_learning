

# the motor model are the actuators from the biped_RL.xml

import numpy as np
import mujoco
# /home/marrodri/Documents/code-repositories/robot_learning_sorina/robot_learning/src/assets/biped/xmls/biped_RL.xml
XML = "../../assets/biped/xmls/biped_RL.xml"
DT_CTRL = 0.01
SIM_DT = 0.001
N_SUB = int(round(DT_CTRL/SIM_DT))
T = 4000 # control steps

model = mujoco.MjModel.from_xml_path(XML)
model.opt.timestep = SIM_DT
data = mujoco.MjData(model)

#Hidden "true" params (example)
for name in ["L_HAA", "L_HFE", "R_HAA", "R_HFE"]:
    a = model.actuator(name)
    a.gainprm[0] = 42.0 # kp
    a.biasprm[1] = -3.6 # -kv
    a.biasprm[2] = -42.0 # -kp
for name in ["L_KFE", "R_KFE"]:
    a = model.actuator(name)
    a.gainprm[0] = 36.0
    a.biasprm[1] = -1.1
    a.biasprm[2] = -36.0

# Initialize at home keyframe
data.qpos[:] = model.keyframe("home").qpos
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

nu, nq, nv = model.nu, model.nq, model.nv
u = np.zeros((T,nu))
q = np.zeros((T, nq))
qd = np.zeros((T,nv))

# Excitation: piecewise-random absolute targets around home ctrl
u0 = model.keyframe("home").ctrl.copy()
rng = np.random.default_rng(0)

for t in range(T):
    if t% 20 == 0:
        delta = rng.uniform(-0.25, 0.25, size=nu) # tune per joint
        cmd = u0 + delta
    u[t] = cmd
    data.ctrl[:] = cmd

    for _ in range(N_SUB):
        mujoco.mj_step(model, data)

    q[t] = data.qpos
    qd[t] = data.qvel

# Add measurement noise + 1-step sensor delay
q_meas = q + rng.normal(0.0, 0.002, size=q.shape)
qd_meas = qd + rng.normal(0.0, 0.02, size = qd.shape)
q_meas[1:] = q_meas[:-1]
qd_meas[1:] = qd_meas[:-1]

np.savez(
    "./results/motor_id_dataset.npz",
    u=u,
    q=q_meas,
    qd=qd_meas,
    dt_ctrl=np.array(DT_CTRL, dtype=np.float64)
)
print("./results/saved motor_id_dataset.npz", u.shape, q_meas.shape, qd_meas.shape)

