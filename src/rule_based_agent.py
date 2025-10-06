import numpy as np
from house_env import HouseEnv
import matplotlib.pyplot as plt

class RuleBasedAgent:
    def __init__(self):
        self.dt = 600          # Time step in seconds (10 minutes)
        self.N_day = 3           # Number of days to simulate
        self.N = self.N_day*24*60*60 // self.dt       # Total number of time steps

        self.T_out_measurement = 10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(self.N_day*24*3600//self.dt)/(24*3600//self.dt)) \
            + np.random.randn(self.N_day*24*3600//self.dt)/5

    def run(self):
        env = HouseEnv(
            T_out_measurement=list(self.T_out_measurement), 
            dt=self.dt,
            render_mode="human", 
            window_size=24*60*60//self.dt,
            owner_schedule=[(72, 108)]
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


if __name__ == "__main__":
    agent = RuleBasedAgent()
    agent.run()
