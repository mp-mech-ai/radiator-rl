from radiator_rl.agents.dqn_agent import DQNAgent
import torch

# Check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Training workflow
agent = DQNAgent(
    hidden_dim=128,     # Size of the hidden dimension in LSTM
    num_layers=2,       # Number of LSTM layers
    output_dim=5,       # 5 level of radiator
    device=device,
    num_workers=1,
    data_path="data/clean/t_out.csv",
    seed=42
    )

number_of_episodes = 2000

# metrics = agent.train(episodes=number_of_episodes)
# agent.save(f"radiator_rl/models/dqn_{number_of_episodes}.pt")

# Evaluation workflow
agent.load(f"radiator_rl/models/dqn_3000_ft.pt")
results = agent.train(episodes=number_of_episodes)

agent.save(f"radiator_rl/models/dqn_5000_ft.pt")
