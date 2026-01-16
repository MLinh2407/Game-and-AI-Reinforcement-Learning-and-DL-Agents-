import os
import config
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
from arena import ArenaEnvironment

os.makedirs("models/hyperparam_tests", exist_ok=True)

def make_env():
    env = ArenaEnvironment(control_scheme=config.CONTROL_ROTATION, render_mode=None)
    env = Monitor(env)
    return env

# Test different learning rates
def test_learning_rates():
    learning_rates = [1e-4, 3e-4, 1e-3]
    
    print("=" * 60)
    print("Testing Learning Rates")
    print("=" * 60)
    
    for lr in learning_rates:
        print(f"\nTraining with LR={lr}...")
        
        env = make_vec_env(make_env, n_envs=1)
        
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.Tanh
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"logs/tensorboard/lr_test",
            verbose=0
        )
        
        # Train for 100K steps
        model.learn(total_timesteps=100000, progress_bar=True)
        model.save(f"models/hyperparam_tests/lr_{lr}")
        
        env.close()
        print(f"✅ Saved model with LR={lr}")

# Test different entropy coefficients
def test_entropy_coefficients():
    entropy_values = [0.0, 0.01, 0.05]
    
    print("\n" + "=" * 60)
    print("Testing Entropy Coefficients")
    print("=" * 60)
    
    for ent in entropy_values:
        print(f"\nTraining with entropy={ent}...")
        
        env = make_vec_env(make_env, n_envs=1)
        
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.Tanh
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"logs/tensorboard/entropy_test",
            verbose=0
        )
        
        model.learn(total_timesteps=100000, progress_bar=True)
        model.save(f"models/hyperparam_tests/entropy_{ent}")
        
        env.close()
        print(f"✅ Saved model with entropy={ent}")

# Test different network sizes
def test_network_sizes():
    network_sizes = [128, 256, 512]
    
    print("\n" + "=" * 60)
    print("Testing Network Sizes")
    print("=" * 60)
    
    for size in network_sizes:
        print(f"\nTraining with network size={size}...")
        
        env = make_vec_env(make_env, n_envs=1)
        
        policy_kwargs = dict(
            net_arch=[dict(pi=[size, size], vf=[size, size])],
            activation_fn=nn.Tanh
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"logs/tensorboard/network_test",
            verbose=0
        )
        
        model.learn(total_timesteps=100000, progress_bar=True)
        model.save(f"models/hyperparam_tests/network_{size}")
        
        env.close()
        print(f"✅ Saved model with network size={size}")

# Test different discount factors
def test_gamma_values():
    gamma_values = [0.95, 0.99]
    
    print("\n" + "=" * 60)
    print("Testing Gamma (Discount Factor)")
    print("=" * 60)
    
    for gamma in gamma_values:
        print(f"\nTraining with gamma={gamma}...")
        
        env = make_vec_env(make_env, n_envs=1)
        
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.Tanh
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"logs/tensorboard/gamma_test",
            verbose=0
        )
        
        model.learn(total_timesteps=150000, progress_bar=True)
        model.save(f"models/hyperparam_tests/gamma_{gamma}")
        
        env.close()
        print(f"✅ Saved model with gamma={gamma}")

if __name__ == "__main__":
    print("Starting Hyperparameter Testing")
    print("This will take approximately 2-3 hours")
    print("=" * 60)
    
    # Run all tests
    test_learning_rates()      
    test_entropy_coefficients() 
    test_network_sizes()       
    test_gamma_values()       
    
    print("\n" + "=" * 60)
    print("✅ All hyperparameter tests completed!")
    print("=" * 60)
    print("\nTo view results in TensorBoard:")
    print("  tensorboard --logdir logs/tensorboard")