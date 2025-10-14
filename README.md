# Reinforcement Learning for Radiator Control: Energy-Efficient Temperature Management

## 1. Overview

This project implements a Reinforcement Learning (RL) approach to control a building's radiator system, balancing energy savings and occupant comfort. The goal is to minimize electricity costs (including off-peak pricing) while maintaining comfortable temperatures when occupants are home.

## 2. Key Features

- **Simulated Environment**: Custom linear physical model (conductance + capacity) for building thermal dynamics.
- **RL Agents**: Rule-based baseline and a DQN (Deep Q-Network) implemented in PyTorch.
- **Discrete Action Space**: Radiator power levels.
- **Reward Function**: Weighted sum of electricity cost and comfort (temperature deviation).
- **Data**: Synthetic data for algorithm testing; real-world data (MeteoSwiss) for training.

## 3. Environment

### State Space

- Current indoor temperature
- Outdoor temperature
- Time of day (for pricing)
- Occupant presence (binary or range)
- Electricity price (real-time or averaged)

### Action Space

- Discrete radiator power levels (e.g., 0%, 33%, 66%, 100%)

### Reward Function

- **Cost Term**: Penalizes high electricity usage, scaled by real-time pricing.
- **Comfort Term**: Penalizes deviation from the desired temperature range.
- **Total Reward**: Weighted sum of cost and comfort terms.

## 4. Data

- **Weather Data**: Sourced from [MeteoSwiss](https://www.meteoswiss.admin.ch/) (real-world) and synthetic datasets (testing).
- **Electricity Pricing**: Simulated off-peak/peak pricing (averaged for training).
- **Occupant Presence**: Simulated presence ranges.

## 5. Algorithms

### Rule-Based Agent

- Simple heuristic (e.g., turn on radiator if temperature is below a threshold).

### DQN Agent

- **Network**: PyTorch implementation.
- **Training**: Offline (pre-collected data) or online (interaction with the environment).
- **Hyperparameters**: Learning rate, discount factor, exploration rate (ε-greedy).

## 6. Evaluation

### Metrics

- **Total Reward**: Sum of rewards over a 24-hour period.
- **Total Cost**: Sum of electricity costs over a 24-hour period.
- **Comfort Metrics**: Average deviation from the desired temperature.

### Baselines

- Rule-based agent (for comparison).

## 7. Setup & Reproducibility

### Dependencies

- Python 3.8+
- PyTorch
- Gymnasium
- Poetry (for dependency management)

### Installation

1. Clone the repository:
   ```
   git clone [your-repo-link]
   ```
2. Install dependencies:
   ```
   poetry install
   ```
3. Download weather data from MeteoSwiss and place it in `data/weather/`.

### Training

- Run the DQN training script:
  ```
  python train.py --config config/dqn.yaml
  ```

### Testing

- Evaluate the trained agent:
  ```
  python evaluate.py --model_path models/dqn.pth
  ```

## 8. Challenges & Limitations

- **Long Training Time**: DQN requires extensive interaction with the environment.
- **Localization**: Model is currently trained for a single location.
- **Temperature Assumptions**: Simplified linear model may not capture all real-world dynamics.

## 9. Future Work

- **Scalability**: Extend to multiple locations with diverse weather patterns.
- **Advanced Algorithms**: Implement PPO for continuous power control.
- **Real-World Deployment**: Test on physical hardware or more complex simulators.

## 10. License

[MIT License](LICENSE)
