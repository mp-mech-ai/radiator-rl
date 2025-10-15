from radiator_rl.agents.dqn_agent import DQNAgent
import torch

# Check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

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

metrics = agent.train(episodes=1000)
agent.save("radiator_rl/models/dqn_1000_normalized.pt")

# Evaluation workflow
agent.load("radiator_rl/models/dqn_1000_normalized.pt")
results = agent.run(episodes=1, render=True, data_index=50)
