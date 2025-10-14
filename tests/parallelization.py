import os
import time
from radiator_rl.agents.dqn_agent import DQNAgent
import torch

"""
Parallelization is only effective if the episode duration is long or if the calculation of each
steps are costly. If on of those conditions aren't met you should consider training sequentially.

In all case, it is recommended to time a single episode with 1 worker and N worker and see if 
it is efficient.
"""

num_workers = len(os.sched_getaffinity(0))
print(f"num_worker: {num_workers}")
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
agent = DQNAgent(
    hidden_dim=128, 
    num_layers=2, 
    output_dim=5,
    device=device,
    data_path="data/clean/t_out.csv",
    num_workers=1
    )
rewards = agent.run(is_training=True, render=False, episodes=5)
print(agent.optimization_steps)
print(f"---Not parallelized: {time.time() - t0:.2f}s ---")

t0 = time.time()
agent = DQNAgent(
    hidden_dim=128, 
    num_layers=2, 
    output_dim=5,
    device=device,
    data_path="data/clean/t_out.csv",
    num_workers=num_workers
    )
rewards = agent.run(is_training=True, render=False, episodes=5)
print(agent.optimization_steps)
print(f"---Parallelized: {time.time() - t0:.2f}s ---")