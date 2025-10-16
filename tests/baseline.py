from radiator_rl.utils import plot_model
from radiator_rl.agents.rule_based_agent import RuleBasedAgent
from tqdm import tqdm
import numpy as np


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

model_data = {
    'time': [f["current_step"] for f in infos_rb],
    'T_in': [f["T_in"] for f in infos_rb],
    'T_out': [f["T_out"] for f in infos_rb],
    'energy_consumed': [f["energy_consumed"] for f in infos_rb],
    'energy_cost': [f["energy_cost"] for f in infos_rb],
    'reward': [f["reward"] for f in infos_rb]
}

# For the rule-based agent
fig, ax = plot_model(
    model_data=model_data,
    dt=600,
    model_name="Rule-Based",
    owner_schedule=[(0, 36), (108, 144)],
    steps_per_day=144,
    show=True
)

num_day_eval = 365
total_rb_reward = 0.0
total_rb_energy = 0.0
total_rb_cost = 0.0

for day in tqdm(range(num_day_eval)):
    infos_rb = rule_based.run(data_index=DATA_INDEX, verbose=False)
    rb_reward = np.array([f["reward"] for f in infos_rb]).sum()
    rb_energy = np.array([f["energy_consumed"] for f in infos_rb]).sum()
    rb_cost = np.array([f["energy_cost"] for f in infos_rb]).sum()

    total_rb_reward += rb_reward
    total_rb_energy += rb_energy
    total_rb_cost += rb_cost

print(f"\n{'='*50}")
print(f"Average Daily Reward:")
print(f"  Rule-Based: {total_rb_reward/num_day_eval:.4f}")
print(f"\nAverage Daily Energy (kWh):")
print(f"  Rule-Based: {total_rb_energy/num_day_eval:.4f}")
print(f"\nAverage Daily Cost (CHF):")
print(f"  Rule-Based: {total_rb_cost/num_day_eval:.4f}")
print(f"{'='*50}")