# This code holds the gymnasium environment for the heating system
# The PDE is C * dT_in/dt = G * (T_out - T_in) + P_radiator

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta

class HouseEnv(gym.Env):
    def __init__(
            self, 
            T_out_measurement : list[float],
            dt: int = 600,
            start_time: str = "2025-01-01 00:00:00", 
            T_in_initial : float = None,
            C: float = 7.10**7,                 # Thermal capacity of the building [J/K]
            G: float = 100,                     # Thermal conductance of the building [W/K]
            radiator_states: int = 2,           # Number of radiator power levels (0: off, 1: level 1, 2: level 2, etc.)
            radiator_factor: float = 2000,      # Radiator factor [W], level 1 corresponds to 2000W, level 2 to 4000W, etc.
            eta: float = 0.9,                   # Efficiency of the heating system 
            owner_schedule: list[tuple] = [(0, 36), (108, 144)],  # Time stamps (in dt steps) when owners come back home (e.g., 36 for 6am, 108 for 18pm with dt=600s)
            off_peak_schedule: list[tuple] = [(0, 36), (132, 144)], # Time stamps (in dt steps) when off-peak hours occurs (20% off to the price)
            kWh_price: float = 0.31,            # 31 Rp/kWh in Vaud: https://visualize.admin.ch/fr/v/qSyFtzDvZlmL?dataSource=Prod
            render_mode = None,                 # "human" for rendering, None for no rendering
            window_size = None,                 # time Window size for rendering
            seed=None
            ):
        
        super().__init__()
        
        # Parameters
        self.C = C  
        self.G = G  
        self.radiator_states = np.arange(radiator_states)  # Radiator power levels [W]
        self.radiator_factor = radiator_factor  # Radiator factor [W]
        self.radiator_powers = [i * radiator_factor for i in self.radiator_states]  # Radiator power levels
        self.eta = eta  # Efficiency of the heating system
        self.lambda_energy = 1.5  # Weight factor for energy consumption in the reward function

        self.time_manager = TimeManager(
            dt=dt, 
            owner_schedule=owner_schedule,
            off_peak_schedule=off_peak_schedule,
            start_time=start_time
            )
        
        self.kWh_price = kWh_price
        self.render_mode = render_mode

        # Normalization constants
        self.T_mean = 10.0
        self.T_std = 15.0

        # Action space: Radiator levels
        self.action_space = Discrete(len(self.radiator_states))

        # Observation space - normalized
        low = np.array([
            -2.0,  # T_in normalized
            -2.0,  # T_out normalized
            0.0,   # radiator state normalized
            0.0,   # time before owners normalized
            0.0    # time of day normalized
        ], dtype=np.float32)
        
        high = np.array([
            2.0,   # T_in normalized
            2.0,   # T_out normalized
            1.0,   # radiator state normalized
            1.0,   # time before owners normalized
            1.0    # time of day normalized
        ], dtype=np.float32)
    
        self.observation_space = Box(low=low, high=high, seed=42)
        self.rng = np.random.default_rng(seed)

        # Initial conditions
        if T_in_initial is None and self.time_manager._is_owner_present():
            self.T_in_initial = 21.0 + (2*self.rng.standard_normal() - 1) # Initial internal temperature [C]
        elif T_in_initial is None and not self.time_manager._is_owner_present():
            self.T_in_initial = T_out_measurement[0] + (2*self.rng.standard_normal() - 1)  # Initial internal temperature [C]
        else:
            self.T_in_initial = T_in_initial  # Initial internal temperature [C]
        self.T_in = self.T_in_initial  # Current internal temperature [C]

        self.T_out_measurement = T_out_measurement  # External temperature profile [C]
        self.T_out = T_out_measurement[0]  # Current external temperature [C]

        self.radiator_state = 0  # Radiator state

        # Stop condition
        self.max_time = len(T_out_measurement) - 1 # Maximum time steps in the simulation

        if self.render_mode == "human":
            if not window_size:
                self.window_size = self.time_manager.steps_per_day  # Default to one day
            else:
                self.window_size = window_size
            self.time_history = deque([self.time_manager.current_step], maxlen=window_size)  # Time history for plotting
            self.T_in_history = deque([T_in_initial], maxlen=window_size)  # T_in history for plotting
            self.T_out_history = deque([T_out_measurement[0]], maxlen=window_size)  # T_out history for plotting
            self.energy_consumption_history = deque([0.], maxlen=window_size)  # Power consumption history for plotting

            self.fig, self.ax = plt.subplots(ncols=2, figsize=(10,5))
            self.line_T_out, = self.ax[0].plot(self.time_history, self.T_out_history, 'b', label='T_out')
            self.line_T_in, = self.ax[0].plot(self.time_history, self.T_in_history, 'r', label='T_in')
            self.line_target = self.ax[0].axhspan(20, 22, color='r', alpha=0.2, label='Target')

            # Add a placeholder patch for owner presence legend
            self.owner_presence_legend = self.ax[0].axvspan(1000, 1000, color='green', alpha=0.1, label='Owner Present')
            self.owner_presence_patches = []

            T_max, T_min = max(25, self.T_out, self.T_in)+5, min(-10, self.T_out, self.T_in)-5
            self.ax[0].set_ylim(T_min, T_max)
            self.ax[0].set_xlabel(f'Time stamp (every {dt/60:.0f} min)')
            self.ax[0].set_ylabel('Temperature (C)')
            self.ax[0].legend(loc='lower left')

            self.line_energy, = self.ax[1].plot(self.time_history, self.energy_consumption_history, 'g', label='Total energy Consumption (kWh)')
            self.ax[1].set_ylim(0, 1.1*self.time_manager.dt.total_seconds()*max(self.radiator_powers)/self.eta)
            self.ax[1].set_xlabel(f'Time stamp (every {dt/60:.0f} min)')
            self.ax[1].set_ylabel('Total Energy Consumption (kWh)')
            self.ax[1].legend(loc='upper left')

            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()

    def _normalize_observation(self, T_in, T_out, radiator_state, time_before, step_of_day):
        """Normalize observations to help with learning."""
        return np.array([
            (T_in - self.T_mean) / self.T_std,
            (T_out - self.T_mean) / self.T_std,
            radiator_state / max(1, (self.action_space.n - 1)),
            time_before / self.time_manager.steps_per_day,
            step_of_day / self.time_manager.steps_per_day
        ], dtype=np.float32)
    
    def _unnormalize_observation(self, normalized_obs):
        """Convert normalized observations back to original scale.
        
        Args:
            normalized_obs: Normalized observation array of shape (5,)
        
        Returns:
            Dictionary with unnormalized values
        """
        return np.array([
            normalized_obs[0] * self.T_std + self.T_mean,
            normalized_obs[1] * self.T_std + self.T_mean,
            int(normalized_obs[2] * (self.action_space.n - 1)),
            int(normalized_obs[3] * self.time_manager.steps_per_day),
            int(normalized_obs[4] * self.time_manager.steps_per_day)
        ])


    def step(self, 
            action: int
            ):
        # Validate action
        if action not in self.radiator_states:
            raise ValueError(f"Invalid action. Action must be a value between {self.radiator_states[0]} and {self.radiator_states[-1]}.")
        
        # Apply action
        self.radiator_state = action
        P_radiator = self.radiator_powers[self.radiator_state]
        radiator_energy_consumption = (self.time_manager.dt.total_seconds() * P_radiator / self.eta ) / (3600*1000)  # in kWh

        # Update internal temperature using Euler method
        dT_in_dt = (self.G * (self.T_out - self.T_in) + P_radiator) / self.C
        new_T_in = self.T_in + dT_in_dt * self.time_manager.dt.total_seconds()
        self.T_in = new_T_in

        # Update time 
        self.time_manager.step()
        self.T_out = self.T_out_measurement[self.time_manager.current_step]
        
        # Normalized observation
        observation = self._normalize_observation(
            self.T_in,
            self.T_out,
            self.radiator_state,
            self.time_manager.time_before_owners_come_back,
            self.time_manager.current_step
        )

        # ----------- Reward calculation -----------------
        if self.time_manager.time_before_owners_come_back == 0:
            temperature_reward = - max(0, abs(self.T_in - 21) - 1)  # Penalize deviation from 21 C beyond a tolerance of 1 C
        else:
            temperature_reward = 0  # No penalty when owners are not home
        
        if not self.time_manager.is_off_peak():
            energy_cost_reward = -self.kWh_price * radiator_energy_consumption
        else:
            energy_cost_reward = -0.8 * self.kWh_price * radiator_energy_consumption
        
        energy_penalty = energy_cost_reward * self.lambda_energy  # Penalize energy consumption

        reward = temperature_reward + energy_penalty

        # Check if episode is done
        terminated = self.time_manager.current_step >= self.max_time
        if terminated:
            self.time_manager.reset()  # Reset time manager for next episode
        truncated = False

        info = {
            "current_step": self.time_manager.current_step,
            "T_in": self.T_in,
            "T_out": self.T_out,
            "energy_consumed": radiator_energy_consumption, 
            "energy_cost": -energy_cost_reward,
            "reward": reward
        }

        if self.render_mode == "human" and not terminated:
            self.time_history.append(self.time_manager.current_step)
            self.T_in_history.append(self.T_in)
            self.T_out_history.append(self.T_out)
            self.energy_consumption_history.append(self.energy_consumption_history[-1] + radiator_energy_consumption)
            self.render()

        return observation, reward, terminated, truncated, info
        
    def render(self):
        # Update plot with current data
        self.line_T_in.set_data(self.time_history, self.T_in_history)
        self.line_T_out.set_data(self.time_history, self.T_out_history)

        T_max, T_min = max(25, *self.T_out_history, *self.T_in_history)+5, min(-10, *self.T_out_history, *self.T_in_history)-5
        self.ax[0].set_ylim(T_min, T_max)
        self.ax[0].set_xlim(min(self.time_history), max(self.time_history)+1)
        self.ax[0].autoscale_view()

        self.line_energy.set_data(self.time_history, self.energy_consumption_history)
        self.ax[1].set_ylim(0, max(1, 1.1*max(self.energy_consumption_history)))
        self.ax[1].set_xlim(min(self.time_history), max(self.time_history)+1)
        self.ax[1].autoscale_view()

        self._update_owner_presence_patches()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        # Initial conditions
        self.T_in = self.T_in_initial  # Initial internal temperature [C]

        self.T_out = self.T_out_measurement[0]  # Current external temperature [C]
        
        self.radiator_state = 0  # Radiator state

        if self.render_mode == "human":
            self.time_history = deque([self.time_manager.current_step]*self.window_size, maxlen=self.window_size)  # Time history for plotting
            self.T_in_history = deque([self.T_in_initial]*self.window_size, maxlen=self.window_size)  # T_in history for plotting
            self.T_out_history = deque([self.T_out_measurement[0]]*self.window_size, maxlen=self.window_size)  # T_out history for plotting
            self.energy_consumption_history = deque([0.]*self.window_size, maxlen=self.window_size)  # Energy consumption history for plotting
        
        # Normalized observation
        observation = self._normalize_observation(
            self.T_in,
            self.T_out,
            self.radiator_state,
            self.time_manager.time_before_owners_come_back,
            self.time_manager.current_step
        )

        return observation, {}
    
    def _update_owner_presence_patches(self):
        """Update axvspan patches to show owner presence periods in the current window."""
        # Remove old patches
        for patch in self.owner_presence_patches:
            patch.remove()
        self.owner_presence_patches.clear()

        # Get current window range
        min_step = min(self.time_history)
        max_step = max(self.time_history)

        # Calculate owner presence periods within the visible window
        presence_periods = []
        for step in range(min_step, max_step + 1):
            step_in_day = step % self.time_manager.steps_per_day
            is_present = any(start <= step_in_day < end for start, end in self.time_manager.owner_schedule)
            
            if is_present:
                if not presence_periods or presence_periods[-1][1] != step - 1:
                    # Start a new period
                    presence_periods.append([step, step])
                else:
                    # Extend the current period
                    presence_periods[-1][1] = step

        # Create axvspan patches for each presence period
        for start, end in presence_periods:
            patch = self.ax[0].axvspan(start, end + 1, color='green', alpha=0.1, 
                                    label='Owner Present' if not self.owner_presence_patches else '')
            self.owner_presence_patches.append(patch)

class TimeManager():
    def __init__(self,
                 dt,
                 owner_schedule=[(0, 36), (108, 144)],
                 off_peak_schedule=[(0, 36), (132, 144)],
                 start_time="2025-01-01 00:00:00"
                 ):
        """
        Manages the simulation time and owner presence schedule. Everythin is in time steps of dt seconds.
        Args:
            dt (int): Time step in seconds.
            owner_schedule (list of tuples): List of (start, end) time steps when owners are present.
            start_time (str): Start time in "YYYY-MM-DD HH:MM:SS" format.
        """
        self.dt = timedelta(seconds=dt)
        self.owner_schedule = owner_schedule
        self.off_peak_schedule = off_peak_schedule
        self.start_time = start_time
        self.current_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.current_step = 0
        self.time_before_owners_come_back = self.owner_schedule[0][0]  # Initial time before owners come back (in time steps)

        # Calculate the number of time steps per day
        self.steps_per_day = int(timedelta(days=1).total_seconds() // self.dt.total_seconds())

    def is_off_peak(self):
        """Returns True if the current time step is within an off-peak period."""
        current_step_in_day = self.current_step % self.steps_per_day
        for start, end in self.off_peak_schedule:
            if start <= current_step_in_day < end:
                return True
        return False
    
    def _is_owner_present(self):
        """Returns True if the current time step is within an owner presence period."""
        current_step_in_day = self.current_step % self.steps_per_day
        for start, end in self.owner_schedule:
            if start <= current_step_in_day < end:
                return True
        return False

    def _update_time_before_return(self):
        """Updates time_before_owners_come_back based on the current position."""
        current_step_in_day = self.current_step % self.steps_per_day

        if self._is_owner_present():
            self.time_before_owners_come_back = 0
        else:
            # Find the next presence period
            next_start = None
            for start, end in self.owner_schedule:
                if current_step_in_day < start:
                    next_start = start
                    break
            if next_start is not None:
                self.time_before_owners_come_back = next_start - current_step_in_day
            else:
                # If we are past all periods, wait for the first period the next day
                self.time_before_owners_come_back = self.steps_per_day - current_step_in_day + self.owner_schedule[0][0]

    def step(self):
        self.current_time += self.dt
        self.current_step += 1
        self._update_time_before_return()

    def reset(self):
        self.current_time = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        self.current_step = 0
        self.time_before_owners_come_back = self.owner_schedule[0][0]
        
    @property
    def hour_of_day(self):
        return self.current_time.hour

    @property
    def weekday(self):
        return self.current_time.weekday()


if __name__ == "__main__":
    timemanager = TimeManager(dt=600, owner_schedule=[(10,36), (108, 144)], start_time="2025-01-01 00:00:00")
    for _ in range(150):
        print(f"Time: {timemanager.current_time}, Hour: {timemanager.hour_of_day}, Weekday: {timemanager.weekday}, Time before owners come back: {timemanager.time_before_owners_come_back} steps")
        timemanager.step()
