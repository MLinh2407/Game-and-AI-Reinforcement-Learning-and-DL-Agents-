import pygame
import sys
import os
import config
import numpy as np
from stable_baselines3 import PPO
from arena import ArenaEnvironment

# Load and evaluate a trained model visually
def evaluate_model(model_path="models/ppo_rotation", num_episodes=5):
    model_file = f"{model_path}.zip"
    if not os.path.exists(model_file):
        print(f"❌ Error: Model file not found: {model_file}")
        print(f"Please train the model first using train_rotation.py")
        sys.exit(1)
    
    print("=" * 60)
    print("Evaluating PPO Agent - Rotation Control Scheme")
    print("=" * 60)
    print(f"Loading model from: {model_file}")
    
    # Load the trained model
    try:
        model = PPO.load(model_path, device="auto")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Create environment with rendering
    env = ArenaEnvironment(control_scheme=config.CONTROL_ROTATION, render_mode='human')
    
    print(f"\nRunning {num_episodes} episodes...")
    print("Controls:")
    print("  - Press 'Q' or close window to quit")
    print("  - Press 'N' to skip to next episode")
    print("=" * 60)
    
    episode_rewards = []
    episode_stats = []
    episode_phases = []
    episode_enemies = []
    episode_spawners = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n✅ Evaluation stopped by user")
                    env.close()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("\n✅ Evaluation stopped by user")
                        env.close()
                        sys.exit(0)
                    elif event.key == pygame.K_n:
                        print(f"Skipping to next episode...")
                        done = True
                        break
            
            if done:
                break
            
            # Get action from the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            env.render()
            
            # Small delay for visual clarity
            pygame.time.delay(10)
        
        episode_rewards.append(total_reward)
        episode_stats.append({
            'phase': info.get('phase', 0),
            'enemies_destroyed': info.get('enemies_destroyed', 0),
            'spawners_destroyed': info.get('spawners_destroyed', 0),
            'player_health': info.get('player_health', 0),
            'steps': step_count
        })
        episode_phases.append(info.get('phase', 0))
        episode_enemies.append(info.get('enemies_destroyed', 0))
        episode_spawners.append(info.get('spawners_destroyed', 0))
        episode_lengths.append(step_count)
        
        print(f"  Episode {episode + 1} finished:")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    Steps: {step_count}")
        print(f"    Phase reached: {episode_stats[-1]['phase']}")
        print(f"    Enemies destroyed: {episode_stats[-1]['enemies_destroyed']}")
        print(f"    Spawners destroyed: {episode_stats[-1]['spawners_destroyed']}")
        print(f"    Final health: {episode_stats[-1]['player_health']}")
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Phase: {np.mean(episode_phases):.2f}")
    print(f"Average Enemies Destroyed: {np.mean(episode_enemies):.2f} ± {np.std(episode_enemies):.2f}")
    print(f"Average Spawners Destroyed: {np.mean(episode_spawners):.2f} ± {np.std(episode_spawners):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate (Phase 3+): {sum(1 for p in episode_phases if p >= 3) / len(episode_phases) * 100:.1f}%")
    print("=" * 60)
    
    env.close()
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    # Allow custom model path via command line argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_rotation"
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    evaluate_model(model_path=model_path, num_episodes=num_episodes)