import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import argparse
from datetime import datetime

def create_env(render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return Monitor(env)

def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.algorithm}_{timestamp}"
    model_dir = f"./models/{args.algorithm}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Lunar Lander Reinforcement Learning Training")
    print(f"{'='*60}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Log Directory: {log_dir}")
    print(f"Model Directory: {model_dir}")
    print(f"{'='*60}\n")
    
    vec_env = make_vec_env("LunarLander-v3", n_envs=args.n_envs)
    
    eval_env = create_env()
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=f"{model_dir}/checkpoints",
        name_prefix=f"{args.algorithm}_lunar_lander",
        verbose=1
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    algorithm_class = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C
    }
    
    if args.algorithm not in algorithm_class:
        raise ValueError(f"Unknown algorithm: {args.algorithm}. Choose from: PPO, DQN, A2C")
    
    AlgoClass = algorithm_class[args.algorithm]
    
    if args.algorithm == "DQN":
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif args.algorithm == "PPO":
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir
        )
    else:
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir
        )
    
    print("Starting training...")
    print("Press Ctrl+C to stop training early (model will be saved)\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    final_model_path = f"{model_dir}/{args.algorithm}_lunar_lander_final"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("\nEvaluating final model...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if mean_reward >= 200:
        print("\n🎉 SUCCESS! The agent has learned to land safely!")
        print("(Score >= 200 is considered a successful landing)")
    elif mean_reward >= 0:
        print("\n📈 PROGRESS! The agent is learning but not yet optimal.")
        print("Consider training for more timesteps.")
    else:
        print("\n⚠️  The agent needs more training.")
        print("Consider increasing timesteps or adjusting hyperparameters.")
    
    vec_env.close()
    eval_env.close()
    
    return model, mean_reward

def main():
    parser = argparse.ArgumentParser(description="Train a Lunar Lander agent using Reinforcement Learning")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "A2C"],
                        help="RL algorithm to use (default: PPO)")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency (default: 10000)")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Checkpoint save frequency (default: 50000)")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
