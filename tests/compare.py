from radiator_rl.agents.dqn_agent import DQNAgent
from radiator_rl.agents.rule_based_agent import RuleBasedAgent
from radiator_rl.utils import compare_models
import torch
import numpy as np
from tqdm import tqdm

DATA_INDEX = 2

# Rule-based agent
rule_based = RuleBasedAgent(
    render=False,
    smartness=1,
    data_path="data/clean/t_out.csv",
    seed=42
)
infos_rb = rule_based.run(data_index=DATA_INDEX, verbose=False)
infos_rb = infos_rb[:-1]

model1_data = {
    'time': [f["current_step"] for f in infos_rb],
    'T_in': [f["T_in"] for f in infos_rb],
    'T_out': [f["T_out"] for f in infos_rb],
    'energy_consumed': [f["energy_consumed"] for f in infos_rb],
    'energy_cost': [f["energy_cost"] for f in infos_rb],
    'temperature_reward': [f["temperature_reward"] for f in infos_rb],
    'reward': [f["total_reward"] for f in infos_rb]
}

# DQN agent
device = "cuda" if torch.cuda.is_available() else "cpu"
dqn = DQNAgent(
    hidden_dim=128,
    num_layers=2,
    output_dim=5,
    device=device,
    data_path="data/clean/t_out.csv",
    num_workers=1,
    seed=42
)

dqn.load("radiator_rl/models/dqn_1000_winter_only.pt")

# Run
infos_dqn = dqn.run(render=False, episodes=1, data_index=DATA_INDEX, verbose=False)
infos_dqn = infos_dqn[:-1]

model2_data = {
    'time': [f["current_step"] for f in infos_dqn],
    'T_in': [f["T_in"] for f in infos_dqn],
    'T_out': [f["T_out"] for f in infos_dqn],
    'energy_consumed': [f["energy_consumed"] for f in infos_dqn],
    'energy_cost': [f["energy_cost"] for f in infos_dqn],
    'temperature_reward': [f["temperature_reward"] for f in infos_dqn],
    'reward': [f["total_reward"] for f in infos_dqn]
}

# Compare
fig, ax = compare_models(
    model1_data=model1_data,
    model2_data=model2_data,
    dt=600,
    model1_name="Rule-Based",
    model2_name="DQN",
    owner_schedule=[(0, 36), (108, 144)],
    steps_per_day=144,
    show=True
)

days_eval = np.concat(
    (np.arange(0, 30*3),np.arange(30*10, 365)),
    axis=0
    )

num_day_eval = len(days_eval)

total_rb_reward = 0.0
total_dqn_reward = 0.0
total_rb_cost = 0.0
total_dqn_cost = 0.0

for day in tqdm(days_eval):
    infos_rb = rule_based.run(data_index=DATA_INDEX, verbose=False)
    rb_reward = np.array([f["temperature_reward"] for f in infos_rb]).sum()
    rb_cost = np.array([f["energy_cost"] for f in infos_rb]).sum()

    total_rb_reward += rb_reward
    total_rb_cost += rb_cost

    infos_dqn = dqn.run(render=False, episodes=1, data_index=DATA_INDEX, verbose=False)
    dqn_reward = np.array([f["temperature_reward"] for f in infos_dqn]).sum()
    dqn_cost = np.array([f["energy_cost"] for f in infos_dqn]).sum()

    total_dqn_reward += dqn_reward
    total_dqn_cost += dqn_cost

print(f"\n{'='*50}")
print(f"Average Daily Reward:")
print(f"  Rule-Based: {total_rb_reward/num_day_eval:.4f}")
print(f"  DQN:        {total_dqn_reward/num_day_eval:.4f}")
print(f"  Improvement: {((total_dqn_reward - total_rb_reward)/abs(total_rb_reward))*100:.2f}%")
print(f"\nAverage Daily Cost (CHF):")
print(f"  Rule-Based: {total_rb_cost/num_day_eval:.4f}")
print(f"  DQN:        {total_dqn_cost/num_day_eval:.4f}")
print(f"  Savings:    {((total_rb_cost - total_dqn_cost)/total_rb_cost)*100:.2f}%")
print(f"{'='*50}")
