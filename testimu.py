#%%
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
#%%
df=pd.read_csv(r"Assets/Walking-1_TS-03155_2026-02-20-12-10-14_aligned.csv")
df2=pd.read_csv(r"Assets/Walking-1_TS-03379_2026-02-20-12-10-14_aligned.csv")

# %%
shank= df[['qx','qy','qz','qr']].values
thigh= df2[['qx','qy','qz','qr']].values

min_len = min(len(thigh), len(shank))

thigh = thigh[:min_len]
shank = shank[:min_len]
# %%
r1=R.from_quat(thigh)

# %%
r2=R.from_quat(shank)

#%%
static_frames= 500

r_thigh_static=r1[:static_frames]
r_shank_static=r2[:static_frames]

r_offset=(r_thigh_static.inv() * r_shank_static).mean()
# %%
r_knee=r1.inv() * r2 
r_knee_corrected=r_knee * r_offset.inv()
# %%
euler_knee=r_knee_corrected.as_euler('xyz', degrees=True)
plt.plot(euler_knee[:,0], label='X-axis')
plt.plot(euler_knee[:,1], label='Y-axis')
plt.plot(euler_knee[:,2], label='Z-axis')
plt.legend()

plt.show()

# %%

root_dir= Path(__file__).resolve().parent
data_dir= root_dir/'Assets'



def load_data():
    trials={}
    for file in data_dir.rglob(r"*.csv"):

        parts= file.stem.split('_')
        if len(parts)==2:
            condition, joint= parts [0],parts[1]
            side='none'
        elif len(parts)==3:
            condition, side, joint= parts [0],parts[1], parts[2]
        else:
            print(f"Skipping unexpected filename: {file.name}")
            continue


        meta_data= pd.read_csv(file)

        quat_cols = ['qx', 'qy', 'qz', 'qr']
        if not all(col in meta_data.columns for col in quat_cols):
            print(f"Skipping {file.name}: missing quaternion columns")
            continue

        quats = meta_data[quat_cols]

        trials.setdefault(condition, {}).setdefault(joint, {})[side] = quats

    return trials
trials=load_data()
#%% Joint function
def calculate_joint_angles(trials, condition, proximal, distal, static_frames=500):
    prox_side, prox_seg = proximal.split('/')
    dist_side, dist_seg = distal.split('/')
    
    q_prox = trials[condition][prox_seg][prox_side][['qx', 'qy', 'qz', 'qr']].values
    q_dist = trials[condition][dist_seg][dist_side][['qx', 'qy', 'qz', 'qr']].values
    
    # trim to same length
    min_len = min(len(q_prox), len(q_dist))
    q_prox = q_prox[:min_len]
    q_dist = q_dist[:min_len]
    
    r_prox = R.from_quat(q_prox)
    r_dist = R.from_quat(q_dist)
    
    # static offset correction
    r_offset = (r_prox[:static_frames].inv() * r_dist[:static_frames]).mean()
    
    # joint angle
    r_joint = r_prox.inv() * r_dist
    r_joint_corrected = r_joint * r_offset.inv()
    
    euler = r_joint_corrected.as_euler('xyz', degrees=True)
    
    return euler

joints = {
    'left_knee':  ('L/thigh',  'L/shank'),
    'right_knee': ('R/thigh', 'R/shank'),
    'left_hip':   ('none/pelvis', 'L/thigh'),
    'right_hip':  ('none/pelvis', 'R/thigh'),
    'left_ankle': ('L/shank',  'L/foot'),
    'right_ankle':('R/shank', 'R/foot'),
}

conditions= ['squat', 'functional','jj','karate','latlunge','lunge','pickup','shoe','sts']

results = {}

for condition in conditions:
    results[condition]={}
    for joint_name, (proximal, distal) in joints.items():
        try:
            results[condition][joint_name] = calculate_joint_angles(
                trials, condition, proximal, distal
                )
        except KeyError:
            print(f"missing data for {condition}-{joint_name},skipping")
results['functional']['right_knee'][:,2] *= -1
results['squat']['right_knee'][:,2] *= -1
results['karate']['right_knee'][:,2] *= -1
results['sts']['right_knee'][:,2] *= -1
results['shoe']['right_knee'][:,2] *= -1


# %%

plt.plot(results['karate']['right_knee'][:,2], label='right Knee flexion(flex+)')
plt.plot(results['karate']['left_knee'][:,2], label= 'left knee flexion(flex+)')
plt.plot(results['karate']['left_hip'][:,1], label='left hip abduction(abd+)')
plt.plot(results['karate']['right_hip'][:,1],label='right hip abduction(abd+)')
plt.legend()
plt.show()
plt.close()
#%%
plt.plot(results['squat']['right_knee'][:,2], label='right knee flexion(flex+)')
plt.plot(results['squat']['left_knee'][:,2],label= 'left knee flexion(flex+)')
plt.legend()
plt.show()
plt.close()
#%%
plt.plot(results['sts']['right_knee'][:,2], label='right Knee flexion(flex+)')
plt.plot(results['sts']['left_knee'][:,2], label= 'left knee flexion(flex+)')
plt.plot(results['sts']['left_hip'][:,1], label='left hip abduction(abd+)')
plt.plot(results['sts']['right_hip'][:,1],label='right hip abduction(abd+)')
plt.legend()
plt.show()
plt.close()
# %%
print(trials.keys())
# %%
plt.plot(results['shoe']['right_knee'][:,2], label='right Knee flexion(flex+)')
plt.plot(results['shoe']['left_knee'][:,2], label= 'left knee flexion(flex+)')
plt.plot(results['shoe']['left_hip'][:,1], label='left hip abduction(abd+)')
plt.plot(results['shoe']['right_hip'][:,1],label='right hip abduction(abd+)')
plt.legend()
plt.ylabel('angle')
plt.xlabel('frames')
plt.show()
plt.close()

# %%
