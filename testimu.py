#%%
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv(r"Assets\Walking-1_TS-03155_2026-02-20-12-10-14_aligned.csv")
df2=pd.read_csv(r"Assets\Walking-1_TS-03379_2026-02-20-12-10-14_aligned.csv")

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
plt.plot(euler_knee)


plt.show()

# %%
