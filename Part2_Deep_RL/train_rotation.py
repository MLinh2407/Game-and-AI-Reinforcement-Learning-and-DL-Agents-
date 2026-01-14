"""
Training script for rotation control scheme using Stable Baselines3 PPO.
Trains an agent with rotation-based controls and saves the model.
"""

import os
import numpy as np
import sys

# Check if tensorboard is installed (required for logging)
try:
    import tensorboard
except ImportError:
    print("❌ Error: TensorBoard is not installed.")
    print("Please install it using: pip install tensorboard")
    sys.exit(1)

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from arena import ArenaEnvironment
import config

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs/tensorboard/rotation", exist_ok=True)
os.makedirs("logs/eval/rotation", exist_ok=True)

def make_env():
    """Create and wrap the environment"""
    env = ArenaEnvironment(control_scheme=config.CONTROL_ROTATION, render_mode=None)
    env = Monitor(env, "logs/eval/rotation")
    return env

def main():
    print("=" * 60)
    print("Training PPO Agent - Rotation Control Scheme")
    print("=" * 60)
    
    # Create vectorized environment (1 environment for training)
    env = make_vec_env(make_env, n_envs=1)
    
    # Define custom policy network architecture
    # Policy network: [256, 256] hidden layers with tanh activation
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=nn.Tanh
    )
    
    # Hyperparameters (tuned for this environment)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,           # Learning rate
        n_steps=2048,                  # Steps per update
        batch_size=64,                 # Batch size
        n_epochs=10,                   # Number of optimization epochs per update
        gamma=0.99,                    # Discount factor
        gae_lambda=0.95,               # GAE lambda parameter
        clip_range=0.2,                # PPO clip range
        ent_coef=0.01,                 # Entropy coefficient (exploration)
        vf_coef=0.5,                   # Value function coefficient
        max_grad_norm=0.5,             # Gradient clipping
        policy_kwargs=policy_kwargs,
        tensorboard_log="logs/tensorboard/rotation",
        verbose=1,
        device="auto"
    )
    
    # Evaluation callback
    eval_env = make_vec_env(make_env, n_envs=1)
    os.makedirs("models/best_rotation", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_rotation/",
        log_path="logs/eval/rotation",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Checkpoint callback (save periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="models/checkpoints/rotation/",
        name_prefix="ppo_rotation"
    )
    
    # Train the model
    total_timesteps = 500000
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"TensorBoard logs: logs/tensorboard/rotation")
    print(f"Model will be saved to: models/ppo_rotation")
    print("\nTo view TensorBoard, run:")
    print("  tensorboard --logdir logs/tensorboard/rotation")
    print("\nTraining started...\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    print(f"\n✅ Training completed!")
    
    # Check if best model exists and use it as final model
    best_model_path = "models/best_rotation/best_model.zip"
    final_model_path = "models/ppo_rotation"
    
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model = PPO.load(best_model_path)
        print(f"Saving best model as final model to {final_model_path}...")
        model.save(final_model_path)
        print(f"✅ Best model saved as final model!")
    else:
        print(f"⚠️ Best model not found, saving last model as final...")
        model.save(final_model_path)
        print(f"✅ Last model saved to: {final_model_path}.zip")

    print(f"✅ Best model (from evaluation) saved to: {best_model_path}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
