import torch
import numpy as np
import yaml
from models.dqn import DQN
from envs.replay_buffer import ReplayBuffer
from envs.house_env import HouseEnv
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
from gymnasium.vector import SyncVectorEnv
from datetime import datetime

class DQNAgent:
    def __init__(
        self,
        hidden_dim,
        num_layers,
        output_dim=5,           # Number of radiator states
        history_length=12,      # 12 time steps history (2 hours with dt=600s)
        hyperparams_path="config/hyperparameters.yml",
        device="cpu",
        data_path=None,  
        num_workers=1
    ):
        hp = self._load_hyperparameters(hyperparams_path)
        self.device = device
        self.learning_rate = hp["learning_rate"]
        self.gamma = hp["gamma"]
        self.batch_size = hp["batch_size"]
        self.epsilon_start = hp["epsilon_start"]
        self.epsilon_end = hp["epsilon_end"]
        self.epsilon_decay = hp["epsilon_decay"]
        self.memory_size = hp["memory_size"]
        self.target_update_freq = hp["target_update_freq"]
        self.history_length = history_length
        self.history = deque(maxlen=history_length)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.policy_dqn = None
        self.target_dqn = None

        self.num_workers = num_workers
        if not data_path:
            N_day = 1                   # Number of days to simulate
            self.dt = 600                    # Time step in seconds (10 minutes)
            self.max_time_step = N_day*24*60*60 // self.dt       # Total number of time steps
            self.T_out_measurement = 10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(N_day*24*3600//self.dt)/(24*3600//self.dt)) \
                 + np.random.randn(N_day*24*3600//self.dt)/5
            self.start_time = "2025-01-01 00:00:00"
        else:
            self.T_out_measurement, self.dt, self.start_time = self._get_T_measurement(data_path)
        
    def get_history_tensor(self):
        """Convert current history deque to a tensor, padding if necessary."""
        history_list = list(self.history)
        
        # If history is not full yet, pad with zeros at the beginning
        if len(history_list) < self.history_length:
            state_dim = history_list[0].shape[0]
            padding_needed = self.history_length - len(history_list)
            padding = [torch.ones(state_dim, device=self.device, dtype=torch.float32)*history_list[0] for _ in range(padding_needed)]
            history_list = padding + history_list
        
        return torch.stack(history_list)
    

    def run(self, 
            is_training=True, 
            render=False,
            episodes=1000
            ):
        
        radiator_factor = 2000 / self.output_dim  # Radiator power factor
        render_mode = "human" if render else None
        
        if self.num_workers == 1:
            env = HouseEnv(
                T_out_measurement=list(self.T_out_measurement), 
                dt=self.dt,
                start_time=self.start_time,
                radiator_states=self.output_dim, 
                radiator_factor=radiator_factor, 
                render_mode=render_mode,
                )
        else:
            envs = SyncVectorEnv(
                lambda: HouseEnv(
                    T_out_measurement=t, 
                    dt=self.dt,
                    start_time=self.start_time,
                    radiator_states=self.output_dim, 
                    radiator_factor=radiator_factor, 
                    render_mode=render_mode,
                    ) for t in self.T_out_measurement
                )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []
        if self.policy_dqn is None or self.target_dqn is None:
            # Initialize new networks
            self.policy_dqn = DQN(
                state_dim=num_states, 
                hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, 
                action_dim=num_actions
                ).to(self.device)
            
            self.target_dqn = DQN(
                state_dim=num_states, 
                hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, 
                action_dim=num_actions
                ).to(self.device)
            
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            self.target_dqn.eval()
        else:
            # Reload pre-trained net
            self.policy_dqn = self.policy_dqn
            self.target_dqn = self.target_dqn

        optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

        if is_training:
            memory = ReplayBuffer(self.memory_size)
            epsilon = self.epsilon_start
        
        # Need to add the parallelisation logic
        for episode in range(episodes):
            self.history.clear()

            state, _ = env.reset()
            state = torch.tensor(state).to(self.device, dtype=torch.float32)

            self.history.append(state)

            episode_reward = 0.0
            done = False

            while not done:
                if render:
                    env.render()
                # Get current history as a tensor
                history_tensor = self.get_history_tensor()

                if is_training and np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values, _ = self.policy_dqn(torch.stack(list(self.history)).unsqueeze(0).to(self.device))
                        action = q_values.argmax().item()

                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward

                next_state = torch.tensor(next_state).to(self.device, dtype=torch.float32)
                
                self.history.append(next_state)
                next_history_tensor = self.get_history_tensor()

                if is_training:
                    memory.push(
                        history_tensor.cpu().numpy(), 
                        action, 
                        reward, 
                        next_history_tensor.cpu().numpy(), 
                        done
                    )

                    if len(memory) >= self.batch_size:
                        self.optimize_model(memory, optimizer)
                    epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)


                state = next_state
            
            reward_per_episode.append(episode_reward)
            print(f"Episode: {episode} Reward: {episode_reward:.2f}")

            # Update target network periodically
            if is_training and episode % self.target_update_freq == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                print(f"Target network updated.")
        if render:
            plt.ioff()
            plt.show()
        env.close()
        return reward_per_episode

    def optimize_model(self, memory, optimizer):
        # Sample a batch of experiences from memory
        batch = memory.sample(self.batch_size)
        state_sequences, actions, rewards, next_state_sequences, dones = batch
        
        # Convert to tensors and move to device
        # state_sequences shape: (batch_size, history_length, state_dim)
        state_sequences = torch.FloatTensor(state_sequences).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_state_sequences = torch.FloatTensor(next_state_sequences).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values from policy network
        # q_values shape: (batch_size, num_actions)
        q_values, _ = self.policy_dqn(state_sequences)
        # Get Q values for the actions that were taken
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values, _ = self.target_dqn(next_state_sequences)
            max_next_q_values = next_q_values.max(1)[0]
            # If episode is done, target is just the reward, otherwise add discounted future reward
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss (Mean Squared Error between current and target Q values)
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize the policy network
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def _load_hyperparameters(self, path, key="house_env1"):
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params[key]

    def _get_T_measurement(self, path):
        df = pd.read_csv(path, index_col=False)
        dt = int((datetime.strptime(df["Date"][1], "%Y-%m-%d %H:%M:%S") - datetime.strptime(df["Date"][0], "%Y-%m-%d %H:%M:%S")).total_seconds())
        total_num_day = len(df) * dt // (3600 * 24)
        step_per_day = 24*3600 // dt

        start_time = df.iloc[0, 0]
        rand_ind = np.random.rand(0, total_num_day, self.num_workers)     # select a subset of days among the total number of days

        T_out_measurement = [[]]*self.num_workers

        for i in rand_ind:
            T_out_measurement[i] = list(df.iloc[i:i+step_per_day, 1])
        
        return T_out_measurement, dt, start_time
        


if __name__ == "__main__":
    num_workers = len(os.sched_getaffinity(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        hidden_dim=128, 
        num_layers=2, 
        output_dim=5,
        device=device,
        data_path="data/clean/t_out.csv",
        num_workers=num_workers
        )

    # rewards = agent.run(is_training=True, render=False, episodes=100)
    rewards = agent.run(is_training=False, render=True, episodes=1)

    # print(rewards)
