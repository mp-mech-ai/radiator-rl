from radiator_rl.agents.dqn_agent import DQNAgent
import torch

# Check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the agent 
agent = DQNAgent(
    hidden_dim=128,     # Size of the hidden dimension in LSTM
    num_layers=2,       # Number of LSTM layers
    output_dim=5,       # 5 level of radiator
    device=device,
    data_path="data/clean/t_out.csv",
    num_workers=1,
    seed=42
    )

# Train for 100 episode
rewards = agent.run(is_training=True, render=False, episodes=100)
print(rewards)
# Display a single episode with the trained agent
r = agent.run(is_training=False, render=True, episodes=1)
print(r)