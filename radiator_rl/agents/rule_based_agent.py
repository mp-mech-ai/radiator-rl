import numpy as np
from radiator_rl.envs.house_env import HouseEnv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from radiator_rl.utils import get_T_measurement

class RuleBasedAgent:
    def __init__(self,
                 render=False,
                 data_path=None,
                 seed=None
                 ):
        self.render = render

        self.dt = 600          # Time step in seconds (10 minutes)
        self.N_day = 3           # Number of days to simulate
        self.N = self.N_day*24*60*60 // self.dt       # Total number of time steps

        self.rng = np.random.default_rng(seed)
        self.data_path = data_path
        if not data_path:
            N_day = 1                   # Number of days to simulate
            self.dt = 600                    # Time step in seconds (10 minutes)
            self.max_time_step = N_day*24*60*60 // self.dt       # Total number of time steps
            self.T_out_measurement = 10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(N_day*24*3600//self.dt)/(24*3600//self.dt)) \
                 + self.rng.standard_normal(N_day*24*3600//self.dt)/5
            self.start_time = "2025-01-01 00:00:00"
        else:
            self.T_out_measurement, self.dt, self.start_time = get_T_measurement(data_path, num_workers=1)
            self.T_out_measurement = self.T_out_measurement[0]

    def run(self, data_index=None, verbose=True):
        render_mode = "human" if self.render else None

        if self.data_path:
            self.T_out_measurement, _, _ = get_T_measurement(self.data_path, data_index=data_index, num_workers=1)
            self.T_out_measurement = self.T_out_measurement[0]

        env = HouseEnv(
            T_out_measurement=list(self.T_out_measurement), 
            dt=self.dt,
            T_in_initial=21,
            render_mode=render_mode, 
            window_size=24*60*60//self.dt,
            )
        infos = []
        # Reset the environment
        observation, _ = env.reset()
        terminated = False
        # Run the environment with a simple rule-based policy
        while not terminated:
            observation = env._unnormalize_observation(observation)
            T_in = observation[0]  # Current indoor temperature
            
            if T_in < 20.5:
                action = 1  # Turn radiator on
            elif T_in > 21.5:
                action = 0  # Turn radiator off
            else:
                action = env.radiator_state  # Keep current state

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            infos.append(info)
            if verbose:
                print(f"Step: {env.time_manager.current_step}/{env.max_time}, T_in: {observation[0]:.2f}, Action: {action}, Reward: {reward:.2f}")
            
            if terminated:
                break
        
        plt.ioff()
        plt.show()
        env.close()
        return infos
    

