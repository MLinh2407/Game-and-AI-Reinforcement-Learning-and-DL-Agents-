import pygame
from arena import ArenaEnvironment
import config

def test_environment(control_scheme='rotation'):
    """Test the environment with random actions"""
    print("🚀 Starting Sci-Fi Arena Environment")
    print("=" * 60)
    print(f"Control Scheme: {control_scheme}")
    print("=" * 60)
    
    # Create environment
    env = ArenaEnvironment(control_scheme=control_scheme, render_mode='human')
    obs, _ = env.reset()
    
    running = True
    episode_count = 0
    total_reward = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if done:
            episode_count += 1
            print(f"Episode {episode_count} finished!")
            print(f"  Phase reached: {info['phase']}")
            print(f"  Enemies destroyed: {info['enemies_destroyed']}")
            print(f"  Spawners destroyed: {info['spawners_destroyed']}")
            print(f"  Total reward: {total_reward:.2f}")
            print("-" * 60)
            
            obs, _ = env.reset()
            total_reward = 0
    
    env.close()
    print("\n✅ Environment closed successfully!")


def test_both_controls():
    """Test both control schemes sequentially"""
    print("\n" + "=" * 60)
    print("Testing ROTATION control first...")
    print("=" * 60)
    test_environment(control_scheme=config.CONTROL_ROTATION)
    
    # Uncomment to test directional control after rotation
    # print("\n" + "=" * 60)
    # print("Testing DIRECTIONAL control...")
    # print("=" * 60)
    # test_environment(control_scheme=config.CONTROL_DIRECTIONAL)


if __name__ == "__main__":
    # Test rotation control
    test_environment(control_scheme='rotation')
    
    # To test directional control instead, use:
    # test_environment(control_scheme='directional')
    
    # To test both sequentially:
    # test_both_controls()