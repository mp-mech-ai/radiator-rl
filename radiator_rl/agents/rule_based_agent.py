import numpy as np
from radiator_rl.envs.house_env import HouseEnv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class RuleBasedAgent:
    def __init__(self,
                 data_path=None,
                 seed=None
                 ):
        self.dt = 600          # Time step in seconds (10 minutes)
        self.N_day = 3           # Number of days to simulate
        self.N = self.N_day*24*60*60 // self.dt       # Total number of time steps
        
        self.rng = np.random.default_rng(seed)
        if not data_path:
            N_day = 1                   # Number of days to simulate
            self.dt = 600                    # Time step in seconds (10 minutes)
            self.max_time_step = N_day*24*60*60 // self.dt       # Total number of time steps
            self.T_out_measurement = 10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(N_day*24*3600//self.dt)/(24*3600//self.dt)) \
                 + self.rng.standard_normal(N_day*24*3600//self.dt)/5
            self.start_time = "2025-01-01 00:00:00"
        else:
            self.T_out_measurement, self.dt, self.start_time = self._get_T_measurement(data_path)

    def run(self):
        env = HouseEnv(
            T_out_measurement=list(self.T_out_measurement), 
            dt=self.dt,
            render_mode="human", 
            window_size=24*60*60//self.dt,
            )

        # Reset the environment
        observation, _ = env.reset()
        terminated = False
        # Run the environment with a simple rule-based policy
        while not terminated:
            T_in = observation[0]  # Current indoor temperature

            # Rule-based policy
            if env.time_manager.time_before_owners_come_back == 0:
                if T_in < 20.5:
                    action = 1  # Turn radiator on
                elif T_in > 21.5:
                    action = 0  # Turn radiator off
                else:
                    action = env.radiator_state  # Keep current state
            else:
                action = 0  # Turn radiator off when owners are away

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {env.time_manager.current_step}/{env.max_time}, T_in: {observation[0]:.2f}, Action: {action}, Reward: {reward:.2f}")
            
            if terminated:
                break
        
        plt.ioff()
        plt.show()
        env.close()
    
    def _get_T_measurement(self, path):
        df = pd.read_csv(path, index_col=False)
        dt = int((datetime.strptime(df["Date"][1], "%Y-%m-%d %H:%M:%S") - datetime.strptime(df["Date"][0], "%Y-%m-%d %H:%M:%S")).total_seconds())
        
        step_per_day = 24*3600 // dt

        start_time = df.iloc[0, 0]
        rand_ind = np.sort(np.random.choice(np.arange(0, len(df) - step_per_day), size=self.num_workers, replace=False))
        
        T_out_measurement = []

        for i, ind in enumerate(rand_ind):
            T_out_measurement.append(list(df.iloc[ind:ind+step_per_day, 1]))
        
        return T_out_measurement, dt, start_time

