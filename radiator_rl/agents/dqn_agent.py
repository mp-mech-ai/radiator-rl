import torch
import numpy as np
import yaml
from radiator_rl.models.dqn import DQN
from radiator_rl.envs.replay_buffer import ReplayBuffer
from radiator_rl.envs.house_env import HouseEnv
from radiator_rl.utils import get_T_measurement
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

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
        num_workers=1,
        seed=None
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

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.policy_dqn = None
        self.target_dqn = None

        self.num_workers = num_workers
        self.rng = np.random.default_rng(seed)
        self.data_path = data_path

        if not data_path:
            N_day = 1
            self.dt = 600
            self.max_time_step = N_day*24*60*60 // self.dt
            self.T_out_measurement = [10 + 5*np.sin(-np.pi/2 + 2*np.pi*np.arange(N_day*24*3600//self.dt)/(24*3600//self.dt)) \
                 + self.rng.standard_normal(N_day*24*3600//self.dt)/5]
            self.start_time = "2025-01-01 00:00:00"
        else:
            self.T_out_measurement, self.dt, self.start_time = get_T_measurement(data_path, num_workers=self.num_workers)
        
        self.optimization_steps = 0
        
    
    def _initialize_networks(self, num_states, num_actions):
        """Initialize policy and target networks."""
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
    
    
    def _get_history_tensor_for_deque(self, history_deque):
        """Convert a history deque to a tensor with padding if necessary."""
        history_list = list(history_deque)
        
        # If history is not full yet, pad with the first observation
        if len(history_list) < self.history_length:
            state_dim = history_list[0].shape[0]
            padding_needed = self.history_length - len(history_list)
            padding = [history_list[0].clone() for _ in range(padding_needed)]
            history_list = padding + history_list
        
        return torch.stack(history_list)
    

    def _create_envs(self, render=False, T_in_initial=None):
        """Create vectorized environments."""
        radiator_factor = 2000 / self.output_dim
        render_mode = "human" if render else None

        if self.num_workers > 1:
            envs = AsyncVectorEnv(
                [lambda t=t: HouseEnv(
                    T_out_measurement=t, 
                    dt=self.dt,
                    start_time=self.start_time,
                    T_in_initial=T_in_initial,
                    radiator_states=self.output_dim, 
                    radiator_factor=radiator_factor, 
                    render_mode=None,
                ) for t in self.T_out_measurement]
            )
        else:
            envs = SyncVectorEnv(
                [lambda t=t: HouseEnv(
                    T_out_measurement=t, 
                    dt=self.dt,
                    start_time=self.start_time,
                    T_in_initial=T_in_initial,
                    radiator_states=self.output_dim, 
                    radiator_factor=radiator_factor, 
                    render_mode=render_mode,
                ) for t in self.T_out_measurement]
            )
        
        return envs


    def train(self, episodes=1000, render=False):
        """
        Train the DQN agent.
        
        Args:
            episodes: Number of training episodes
            render: Whether to render the environment
            
        Returns:
            Dictionary containing training metrics (rewards per episode, losses, epsilon history)
        """
        # Get all available temperature data
        if self.data_path:
            df = pd.read_csv(self.data_path, index_col=False)
            step_per_day = 24*3600 // self.dt
            max_start_index = len(df) - step_per_day
            envs = self._create_envs(render=render)
        
        num_states = envs.observation_space.shape[1]
        num_actions = envs.action_space.nvec[0]

        # Initialize networks if needed
        if self.policy_dqn is None or self.target_dqn is None:
            self._initialize_networks(num_states, num_actions)

        optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
        memory = ReplayBuffer(self.memory_size)
        epsilon = self.epsilon_start
        
        reward_per_episode = []
        epsilon_history = []
        
        for episode in range(episodes):
            if self.data_path:
                random_indices = self.rng.choice(np.arange(0, max_start_index), 
                                            size=self.num_workers, replace=False)
                self.T_out_measurement = []
                for idx in random_indices:
                    self.T_out_measurement.append(list(df.iloc[idx:idx+step_per_day, 1]))
            
            envs = self._create_envs(render=render)
            # Reset all environments
            states, _ = envs.reset()
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            
            # Initialize history for each environment
            histories = [deque(maxlen=self.history_length) for _ in range(self.num_workers)]
            for i in range(self.num_workers):
                histories[i].append(states[i])
            
            episode_rewards = np.zeros(self.num_workers)
            dones = np.zeros(self.num_workers, dtype=bool)
            
            while not dones.all():
                if render:
                    envs.render()
                
                # Select actions for each environment (epsilon-greedy)
                actions = []
                for i in range(self.num_workers):
                    if not dones[i]:
                        if np.random.rand() < epsilon:
                            action = envs.single_action_space.sample()
                        else:
                            with torch.no_grad():
                                history_tensor = self._get_history_tensor_for_deque(histories[i]).unsqueeze(0)
                                q_values, _ = self.policy_dqn(history_tensor)
                                action = q_values.argmax().item()
                    else:
                        action = 0  # Dummy action for finished environments
                    actions.append(action)
                
                # Step all environments
                next_states, rewards, dones_step, _, info = envs.step(np.array(actions))
                next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                
                # Process each environment's transition
                for i in range(self.num_workers):
                    if not dones[i]:
                        # Get history tensors before updating
                        history_tensor = self._get_history_tensor_for_deque(histories[i])
                        # Update history with next state
                        histories[i].append(next_states[i])
                        next_history_tensor = self._get_history_tensor_for_deque(histories[i])
                        
                        # Accumulate reward
                        episode_rewards[i] += rewards[i]
                        
                        # Store transition in replay buffer
                        memory.push(
                            history_tensor.cpu().numpy(),
                            actions[i],
                            rewards[i],
                            next_history_tensor.cpu().numpy(),
                            dones_step[i]
                        )
                        
                        # Mark as done if environment terminated
                        if dones_step[i]:
                            dones[i] = True

                # Optimize model
                if len(memory) >= self.batch_size:
                    self.optimize_model(memory, optimizer)
                    self.optimization_steps += 1

                    # Update target network periodically
                    if self.optimization_steps % self.target_update_freq == 0:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                
                # Decay epsilon
                epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            
            # Log episode results
            avg_episode_reward = np.mean(episode_rewards)
            reward_per_episode.append(avg_episode_reward)
            epsilon_history.append(epsilon)
            
            print(f"Episode: {episode} Avg Reward: {avg_episode_reward:.2f} " 
                  f"Epsilon: {epsilon:.4f} "
                  f"(Min: {episode_rewards.min():.2f}, Max: {episode_rewards.max():.2f})")

        if render:
            plt.ioff()
            plt.show()
        envs.close()

        return {
            'rewards': reward_per_episode,
            'epsilon_history': epsilon_history,
            'final_epsilon': epsilon
        }


    def run(self, episodes=1, render=False, data_index=0, verbose=True):
        """
        Run the trained agent in evaluation mode (no exploration, no training).
        
        Args:
            episodes: Number of episodes to run
            render: Whether to render the environment
            
        Returns:
            List of episode information dictionaries
        """
        if self.policy_dqn is None:
            raise ValueError("No trained model found. Please train the agent first or load weights.")
        
        self.policy_dqn.eval()  # Set to evaluation mode
        if self.data_path:
            self.T_out_measurement, _, _ = get_T_measurement(self.data_path, data_index=data_index, num_workers=self.num_workers)
        envs = self._create_envs(render=render, T_in_initial=21)
        
        all_episode_infos = []
        reward_per_episode = []
        
        for episode in range(episodes):
            infos = []
            # Reset all environments
            states, _ = envs.reset()
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            
            # Initialize history for each environment
            histories = [deque(maxlen=self.history_length) for _ in range(self.num_workers)]
            for i in range(self.num_workers):
                histories[i].append(states[i])
            
            episode_rewards = np.zeros(self.num_workers)
            dones = np.zeros(self.num_workers, dtype=bool)
            
            while not dones.all():
                if render:
                    envs.render()
                
                # Select actions greedily (no exploration)
                actions = []
                for i in range(self.num_workers):
                    if not dones[i]:
                        with torch.no_grad():
                            history_tensor = self._get_history_tensor_for_deque(histories[i]).unsqueeze(0)
                            q_values, _ = self.policy_dqn(history_tensor)
                            action = q_values.argmax().item()
                    else:
                        action = 0  # Dummy action for finished environments
                    actions.append(action)
                
                # Step all environments
                next_states, rewards, dones_step, _, info = envs.step(np.array(actions))
                infos.append(info)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                
                # Process each environment's transition
                for i in range(self.num_workers):
                    if not dones[i]:
                        # Update history with next state
                        histories[i].append(next_states[i])
                        
                        # Accumulate reward
                        episode_rewards[i] += rewards[i]
                        
                        # Mark as done if environment terminated
                        if dones_step[i]:
                            dones[i] = True
            
            # Log episode results
            avg_episode_reward = np.mean(episode_rewards)
            reward_per_episode.append(avg_episode_reward)
            all_episode_infos.append(infos)
            
            if verbose:
                print(f"Episode: {episode} Avg Reward: {avg_episode_reward:.2f} " 
                        f"(Min: {episode_rewards.min():.2f}, Max: {episode_rewards.max():.2f})")

        if render:
            plt.ioff()
            plt.show()
        envs.close()
        
        self.policy_dqn.train()  # Set back to training mode

        return infos


    def optimize_model(self, memory, optimizer):
        """Optimize the policy network using a batch from replay buffer."""
        # Sample a batch of experiences from memory
        batch = memory.sample(self.batch_size)
        state_sequences, actions, rewards, next_state_sequences, dones = batch
        
        # Convert to tensors and move to device
        state_sequences = torch.FloatTensor(state_sequences).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_state_sequences = torch.FloatTensor(next_state_sequences).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values from policy network
        q_values, _ = self.policy_dqn(state_sequences)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values, _ = self.target_dqn(next_state_sequences)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize the policy network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_dqn.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    

    def save(self, path):
        """
        Save the trained model weights.
        
        Args:
            path: Path to save the model (e.g., 'models/dqn_agent.pt')
        """
        if self.policy_dqn is None:
            raise ValueError("No model to save. Train the agent first.")
        
        torch.save({
            'policy_state_dict': self.policy_dqn.state_dict(),
            'target_state_dict': self.target_dqn.state_dict(),
            'optimization_steps': self.optimization_steps,
        }, path)
        print(f"Model saved to {path}")
    

    def load(self, path):
        """
        Load trained model weights.
        
        Args:
            path: Path to the saved model
        """
        # Create a dummy environment to get state/action dimensions
        envs = self._create_envs(render=False, T_in_initial=21)
        num_states = envs.observation_space.shape[1]
        num_actions = envs.action_space.nvec[0]
        envs.close()
        
        # Initialize networks if they don't exist
        if self.policy_dqn is None or self.target_dqn is None:
            self._initialize_networks(num_states, num_actions)
        
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_dqn.load_state_dict(checkpoint['policy_state_dict'])
        self.target_dqn.load_state_dict(checkpoint['target_state_dict'])
        self.optimization_steps = checkpoint.get('optimization_steps', 0)
        
        print(f"Model loaded from {path}")
    

    def _load_hyperparameters(self, path, key="house_env1"):
        """Load hyperparameters from YAML file."""
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params[key]
