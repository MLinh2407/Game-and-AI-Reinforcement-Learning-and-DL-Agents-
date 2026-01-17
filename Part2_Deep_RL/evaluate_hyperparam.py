import numpy as np
from stable_baselines3 import PPO
from arena import ArenaEnvironment
import config

# Quickly evaluate a model without rendering
def quick_evaluate(model_path, n_episodes=5):
    model = PPO.load(model_path)
    env = ArenaEnvironment(control_scheme=config.CONTROL_ROTATION, render_mode=None)
    
    rewards = []
    phases = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        phases.append(info.get('phase', 0))
    
    env.close()
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_phase': np.mean(phases)
    }

# Evaluate all hyperparameter test models
def evaluate_all_hyperparameters():
    print("\n" + "=" * 70)
    print("HYPERPARAMETER EVALUATION RESULTS")
    print("=" * 70)
    
    # Learning Rate Tests
    print("\n### LEARNING RATE COMPARISON ###")
    print(f"{'LR':<10} {'Avg Reward':<15} {'Avg Phase':<12}")
    print("-" * 40)
    
    for lr in [1e-4, 3e-4, 1e-3]:
        try:
            results = quick_evaluate(f"models/hyperparam_tests/lr_{lr}")
            print(f"{lr:<10} {results['avg_reward']:>6.1f} ± {results['std_reward']:<5.1f} {results['avg_phase']:>6.1f}")
        except:
            print(f"{lr:<10} Model not found")
    
    # Entropy Tests
    print("\n### ENTROPY COEFFICIENT COMPARISON ###")
    print(f"{'Entropy':<10} {'Avg Reward':<15} {'Avg Phase':<12}")
    print("-" * 40)
    
    for ent in [0.0, 0.01, 0.05]:
        try:
            results = quick_evaluate(f"models/hyperparam_tests/entropy_{ent}")
            print(f"{ent:<10} {results['avg_reward']:>6.1f} ± {results['std_reward']:<5.1f} {results['avg_phase']:>6.1f}")
        except:
            print(f"{ent:<10} Model not found")
    
    # Network Size Tests
    print("\n### NETWORK SIZE COMPARISON ###")
    print(f"{'Size':<10} {'Avg Reward':<15} {'Avg Phase':<12}")
    print("-" * 40)
    
    for size in [128, 256, 512]:
        try:
            results = quick_evaluate(f"models/hyperparam_tests/network_{size}")
            print(f"{size:<10} {results['avg_reward']:>6.1f} ± {results['std_reward']:<5.1f} {results['avg_phase']:>6.1f}")
        except:
            print(f"{size:<10} Model not found")
    
    # Gamma Tests
    print("\n### DISCOUNT FACTOR (GAMMA) COMPARISON ###")
    print(f"{'Gamma':<10} {'Avg Reward':<15} {'Avg Phase':<12}")
    print("-" * 40)
    
    for gamma in [0.95, 0.99]:
        try:
            results = quick_evaluate(f"models/hyperparam_tests/gamma_{gamma}")
            print(f"{gamma:<10} {results['avg_reward']:>6.1f} ± {results['std_reward']:<5.1f} {results['avg_phase']:>6.1f}")
        except:
            print(f"{gamma:<10} Model not found")
    
    print("\n" + "=" * 70)
    print("Copy these results into your report!")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_all_hyperparameters()