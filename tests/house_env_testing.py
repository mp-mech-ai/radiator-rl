import numpy as np
from radiator_rl.envs.house_env import HouseEnv
import matplotlib.pyplot as plt


rng = np.random.default_rng(1)
N_day = 1
dt = 600
max_time_step = N_day*24*60*60 // dt
T_out_measurement = 10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(N_day*24*3600//dt)/(24*3600//dt)) \
        + rng.standard_normal(N_day*24*3600//dt)/5
start_time = "2025-01-01 00:00:00"

env = HouseEnv(
    T_out_measurement=T_out_measurement,
    dt=dt,
    start_time=start_time,
    T_in_initial=21,
    render_mode="human"
)
env.reset()

done = False
while not done:
    _, _, done, _, _ = env.step(0)

plt.ioff()
plt.show()