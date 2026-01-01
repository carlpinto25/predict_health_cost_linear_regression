# Lunar Lander Reinforcement Learning Project

Train an AI agent to land a spacecraft on the moon using Deep Reinforcement Learning.

## Overview

This project uses the OpenAI Gymnasium `LunarLander-v3` environment with Stable Baselines3 to train RL agents. The goal is to land the spacecraft safely between the flags on the landing pad.

### Environment Details

- **State Space**: 8 continuous values (position, velocity, angle, leg contact)
- **Action Space**: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)
- **Reward**: +100-140 for landing, -100 for crashing, fuel consumption penalties
- **Success**: Score >= 200 is considered a successful landing

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --algorithm PPO --timesteps 500000
python train.py --algorithm DQN --timesteps 500000
python train.py --algorithm A2C --timesteps 500000
python train.py --algorithm PPO --timesteps 1000000 --learning-rate 0.0001 --n-envs 8
```

### Training Arguments

- `--algorithm`: RL algorithm (PPO, DQN, A2C)
- `--timesteps`: Total training steps
- `--learning-rate`: Learning rate
- `--n-envs`: Parallel environments
- `--eval-freq`: Evaluation frequency
- `--checkpoint-freq`: Checkpoint save frequency

## Demo

Watch your trained agent play:

```bash
python demo.py
python demo.py --algorithm PPO --episodes 10
python demo.py --model-path ./models/PPO_20240101_120000/best_model/best_model.zip
```

## Monitoring with TensorBoard

```bash
tensorboard --logdir ./logs
```

## Project Structure

```
lunar/
├── train.py
├── demo.py
├── requirements.txt
├── README.md
├── logs/
└── models/
    └── PPO_YYYYMMDD_HHMMSS/
        ├── best_model/
        ├── checkpoints/
        └── PPO_lunar_lander_final.zip
```

## Expected Results

| Algorithm | Timesteps | Expected Reward |
| --------- | --------- | --------------- |
| PPO       | 500,000   | ~250-280        |
| DQN       | 500,000   | ~200-250        |
| A2C       | 500,000   | ~220-260        |

PPO typically learns fastest and most reliably for this environment.

## License

MIT License
