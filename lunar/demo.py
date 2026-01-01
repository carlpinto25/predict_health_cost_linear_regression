import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import argparse
import os
import glob
import time

def find_latest_model(algorithm="PPO"):
    model_dirs = glob.glob(f"./models/{algorithm}_*")
    if not model_dirs:
        return None
    
    latest_dir = max(model_dirs, key=os.path.getmtime)
    
    best_model = os.path.join(latest_dir, "best_model", "best_model.zip")
    if os.path.exists(best_model):
        return best_model
    
    final_model = os.path.join(latest_dir, f"{algorithm}_lunar_lander_final.zip")
    if os.path.exists(final_model):
        return final_model
    
    checkpoints = glob.glob(os.path.join(latest_dir, "checkpoints", "*.zip"))
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    
    return None

def demo(args):
    algorithm_class = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C
    }
    
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_latest_model(args.algorithm)
        if not model_path:
            print(f"No trained model found for {args.algorithm}!")
            print("Please train a model first using: python train.py")
            return
    
    print(f"\n{'='*60}")
    print(f"Lunar Lander Demo")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*60}\n")
    
    AlgoClass = algorithm_class[args.algorithm]
    model = AlgoClass.load(model_path)
    
    env = gym.make("LunarLander-v3", render_mode="human")
    
    total_rewards = []
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
            
            time.sleep(0.01)
        
        total_rewards.append(episode_reward)
        status = "✅ LANDED!" if episode_reward >= 200 else "❌ CRASHED" if episode_reward < 0 else "⚠️  PARTIAL"
        print(f"Episode {episode + 1}/{args.episodes}: Reward = {episode_reward:.2f} {status}")
    
    env.close()
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n{'='*60}")
    print(f"Demo Complete!")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Successful Landings: {sum(1 for r in total_rewards if r >= 200)}/{args.episodes}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Demo a trained Lunar Lander agent")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "A2C"],
                        help="RL algorithm used (default: PPO)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (auto-detects if not specified)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    
    args = parser.parse_args()
    demo(args)

if __name__ == "__main__":
    main()
