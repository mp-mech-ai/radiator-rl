# This code holds the gymnasium environment for the heating system
# The PDE is C * dT_in/dt = G * (T_out - T_in) + P_radiator

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class HouseEnv(gym.Env):
    def __init__(
            self, 
            T_out_measurement : list,
            dt: int = 600,
            T_in_initial : float = 21.0,
            C: float = 7.10**7, 
            G: float = 100,
            radiator_level: list = [0, 2000],   # Radiator power levels [W]
            eta: float = 0.9,
            render_mode: str = None,
            window_size: int = 50
            ):
        
        super().__init__()
        
        # Parameters
        self.C = C  # Thermal capacity of the building [J/K]
        self.G = G  # Thermal conductance of the building [W/K]
        self.radiator_level = radiator_level  # Radiator power levels [W]
        self.eta = eta  # Efficiency of the heating system
        self.dt = dt  # Time step [s]
        self.render_mode = render_mode

        # Action space: Radiator levels
        self.action_space = Discrete(len(radiator_level))

        # Observation space
        low = np.concatenate((
            [-10.]*2,       # T_in and T_out
            [0.],           # radiator state
            [0.],           # time before owners at home (minutes)
            [0.],           # weekday
            [0.]            # hour of the day
            ), dtype=np.float32
        )
        high = np.concatenate((
            [40]*2,                 # T_in and T_out
            [self.action_space.n],  # radiator state
            [23*60],                # time before owners at home (minutes)
            [6.],                   # weekday
            [23.]                   # hour of the day
            ), dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, seed=42)

    
        # Initial conditions
        self.T_in_initial = T_in_initial  # Initial internal temperature [C]
        self.T_in = T_in_initial  # Current internal temperature [C]

        self.T_out_measurement = T_out_measurement  # External temperature profile [C]
        self.T_out = T_out_measurement[0]  # Current external temperature [C]

        self.radiator_state = 0  # Radiator state {0: off, 1: on}
        self.time_before_owners_come_back = 0  # Time before owners come back home [minutes]
        self.weekday = 0  # Day of the week {0: Monday, ..., 6: Sunday}
        self.hour_of_day = 0  # Hour of the day {0, ..., 23}
        self.current_time = 0  # Current time in dt steps

        # Stop condition
        self.max_time = len(T_out_measurement) - 1 # Maximum time steps in the simulation

        if self.render_mode == "human":
            self.window_size = window_size
            self.time_history = deque([0]*window_size, maxlen=window_size)  # Time history for plotting
            self.T_in_history = deque([T_in_initial]*window_size, maxlen=window_size)  # T_in history for plotting
            self.T_out_history = deque([T_out_measurement[0]]*window_size, maxlen=window_size)  # T_out history for plotting

            self.fig, self.ax = plt.subplots()
            self.line_T_out, = self.ax.plot(self.time_history, self.T_out_history, 'b', label='T_out')
            self.line_T_in, = self.ax.plot(self.time_history, self.T_in_history, 'r', label='T_in')
            self.line_target = self.ax.axhspan(20, 22, color='r', alpha=0.2, label='Target')

            T_max, T_min = max(25, self.T_out, self.T_in)+5, min(-10, self.T_out, self.T_in)-5
            self.ax.set_ylim(T_min, T_max)
            self.ax.set_xlabel(f'Time stamp (every {dt/60:.0f} min)')
            self.ax.set_ylabel('Temperature (C)')
            self.ax.legend(loc='lower left')
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()
    
    def step(self, 
            action: int):
        if action not in [0, 1]:
            raise ValueError("Invalid action. Action must be 0 (off) or 1 (on).")
        
        self.radiator_state = action
        if self.radiator_state == 1:
            P_radiator = 2000  # Radiator power when on [W]
        else:
            P_radiator = 0  # Radiator power when off [W]
        
        # Update internal temperature using Euler method
        dT_in_dt = (self.G * (self.T_out - self.T_in) + P_radiator) / self.C
        new_T_in = self.T_in + dT_in_dt * self.dt
        self.T_in = new_T_in

        # Update time and related variables
        self.current_time += 1
        self.T_out = self.T_out_measurement[self.current_time]

        # Update hour of day and weekday
        if self.current_time % 6 == 0:
            self.hour_of_day = (self.hour_of_day + 1) % 24
            if self.hour_of_day == 0:
                self.weekday = (self.weekday + 1) % 7
        
        # Construct observation
        observation = np.concatenate((
            [self.T_in],                        # T_in
            [self.T_out],                       # T_out
            [self.radiator_state],
            [self.time_before_owners_come_back],
            [self.weekday],
            [self.hour_of_day]
        ))

        # Calculate reward
        reward = - max(0, abs(self.T_in - 21) - 1)  # Penalize deviation from 21C beyond a tolerance of 1C

        # Check if episode is done
        terminated = self.current_time >= self.max_time
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.time_history.append(self.current_time)
            self.T_in_history.append(self.T_in)
            self.T_out_history.append(self.T_out)
            self.render()

        return observation, reward, terminated, truncated, info
        
    def render(self):
        # Update plot with current data
        self.line_T_in.set_data(self.time_history, self.T_in_history)
        self.line_T_out.set_data(self.time_history, self.T_out_history)
        T_max, T_min = max(25, *self.T_out_history, *self.T_in_history)+5, min(-10, *self.T_out_history, *self.T_in_history)-5
        self.ax.set_ylim(T_min, T_max)
        self.ax.set_xlim(min(self.time_history), max(self.time_history)+1)
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def reset(self):
        # Initial conditions
        self.T_in = self.T_in_initial  # Initial internal temperature [C]

        self.T_out = self.T_out_measurement[0]  # Current external temperature [C]
        
        self.radiator_state = 0  # Radiator state {0: off, 1: on}
        self.time_before_owners_come_back = 0  # Time before owners come back home [minutes]
        self.weekday = 0  # Day of the week {0: Monday, ..., 6: Sunday}
        self.hour_of_day = 0  # Hour of the day {0, ..., 23}
        self.current_time = 0  # Current time in dt steps

        if self.render_mode == "human":
            self.time_history = deque([0]*self.window_size, maxlen=self.window_size)  # Time history for plotting
            self.T_in_history = deque([self.T_in_initial]*self.window_size, maxlen=self.window_size)  # T_in history for plotting
            self.T_out_history = deque([self.T_out_measurement[0]]*self.window_size, maxlen=self.window_size)  # T_out history for plotting
        
        # Construct the initial observation
        observation = np.concatenate((
            [self.T_in],  # Last 12 T_in values (2h history)
            [self.T_out], # Last 12 T_out values (2h history)
            [self.radiator_state],
            [self.time_before_owners_come_back],
            [self.weekday],
            [self.hour_of_day]
        ))

        return observation, {}
