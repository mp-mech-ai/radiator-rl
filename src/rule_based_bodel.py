import numpy as np
from house_env import HouseEnv
import matplotlib.pyplot as plt

# Example usage
dt = 600          # Time step in seconds (10 minutes)
N_day = 3           # Number of days to simulate
N = N_day*24*60*60 // dt       # Total number of time steps

T_out_measurement = 10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(N_day*24*3600//dt)/(24*3600//dt)) + np.random.randn(N_day*24*3600//dt)/5

env = HouseEnv(
    T_out_measurement=T_out_measurement, 
    dt=dt,
    T_in_initial=21.0,
    render_mode="human", 
    window_size=24*60//10
    )

# Reset the environment
observation, _ = env.reset()
terminated = False
# Run the environment with a simple rule-based policy
while not terminated:
    T_in = observation[0]  # Current indoor temperature

    # Rule-based policy
    if T_in < 20.5:
        action = 1  # Turn radiator on
    elif T_in > 21.5:
        action = 0  # Turn radiator off
    else:
        action = env.radiator_state  # Keep current state

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step: {env.current_time}/{env.max_time}, T_in: {observation[0]:.2f}, Action: {action}, Reward: {reward:.2f}")
    if terminated:
        break

plt.ioff()
plt.show()
env.close()
