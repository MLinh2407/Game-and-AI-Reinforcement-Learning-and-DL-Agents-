import pygame
import random
import os
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import *
from levels import LEVELS
from gridworld import GridWorld
from assets import load_tiles
from rendering import render
from agents import QLearningAgent, SarsaAgent
from ui import create_ui

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def draw_text(screen, text, x, y, size=18, color=(230, 230, 230)):
    font = pygame.font.SysFont("consolas", size)
    screen.blit(font.render(text, True, color), (x, y))

# Save a training curve showing episode rewards and steps over time
def save_training_curve(rewards, episode_lengths, algo, level_id, intrinsic_suffix=""):
    if len(rewards) < 2 or len(episode_lengths) < 2:
        return

    window = 50
    os.makedirs("plots", exist_ok=True)

    episodes = list(range(1, len(rewards) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # Rewards subplot
    ax1.plot(episodes, rewards, label="Episode reward", alpha=0.7, linewidth=1)

    if len(rewards) >= window:
        smooth_rewards = [
            sum(rewards[max(0, i - window):i + 1]) /
            (i - max(0, i - window) + 1)
            for i in range(len(rewards))
        ]
        ax1.plot(episodes, smooth_rewards, label="Avg reward", linewidth=2)

    ax1.set_ylabel("Reward")
    ax1.set_title(f"{algo} – Level {level_id}{intrinsic_suffix}")
    ax1.legend()

    # Steps subplot
    ax2.plot(episodes, episode_lengths, label="Episode steps", color="orange", alpha=0.7, linewidth=1)

    if len(episode_lengths) >= window:
        smooth_steps = [
            sum(episode_lengths[max(0, i - window):i + 1]) /
            (i - max(0, i - window) + 1)
            for i in range(len(episode_lengths))
        ]
        ax2.plot(episodes, smooth_steps, label="Avg steps", color="red", linewidth=2)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()

    fig.tight_layout()

    path = f"plots/{algo}_level{level_id}{intrinsic_suffix}_training.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"[Saved training curve] {path}")

# Compute the maximum achievable environment reward for a level
def compute_level_max_env_reward(level_grid):
    total = 0
    for row in level_grid:
        for tile in row:
            if tile == APPLE:
                total += 1
            elif tile == CHEST:
                total += 2
    return total

# --------------------------------------------------
# Main application
# --------------------------------------------------
def main():
    pygame.init()
    random.seed(config.SEED)

    # Compute grid and UI panel dimensions
    TILE_W = len(LEVELS[0][0]) * TILE_SIZE
    TILE_H = len(LEVELS[0]) * TILE_SIZE
    PANEL_W = 420

    screen = pygame.display.set_mode((TILE_W + PANEL_W, TILE_H))
    pygame.display.set_caption("Gridworld - Classical RL")

    # Load sprites and rendering assets
    tiles, monsters, agent_sprite = load_tiles()
    game_surface = pygame.Surface((TILE_W, TILE_H))

    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    # Create UI buttons for algorithms, levels, and controls
    buttons, level_buttons = create_ui(TILE_W)

    # Initial state
    level_id = 0
    algo_name = "Q_Learning"
    use_intrinsic = False
    
    EPISODES = config.EPISODES_PER_LEVEL.get(level_id, config.DEFAULT_EPISODES)
    EPSILON_DECAY = int(0.80 * EPISODES)
    
    agent = QLearningAgent(EPSILON_DECAY, intrinsic_reward=use_intrinsic)

    # Initialize environment and agent state
    env = GridWorld(LEVELS[level_id])
    level_max_env_reward = compute_level_max_env_reward(LEVELS[level_id])
    state = env.reset()
    action = agent.select_action(state)

    paused = True
    fast_mode = False
    training_done = False

    episode_reward = 0
    episode_intrinsic_reward = 0 
    rewards = []
    intrinsic_rewards = []
    episode_lengths = []
    steps = 0
    
    running = True

    # --------------------------------------------------
    # Main event and training loop
    # --------------------------------------------------
    while running:
        clock.tick(config.FPS_FAST if fast_mode else config.FPS_VISUAL)
        mouse_pos = pygame.mouse.get_pos()

        # update hover states
        for b in buttons.values():
            b.update_hover(mouse_pos)
        for b in level_buttons:
            b.update_hover(mouse_pos)

        # Event handling
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                suffix = "_intrinsic" if use_intrinsic else ""
                save_training_curve(rewards, episode_lengths, algo_name, level_id, suffix)
                running = False

            if e.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                # Algorithm selection
                if buttons["q"].clicked(pos):
                    algo_name = "Q_Learning"
                if buttons["s"].clicked(pos):
                    algo_name = "SARSA"

                if buttons["q"].clicked(pos) or buttons["s"].clicked(pos):
                    EPISODES = config.EPISODES_PER_LEVEL.get(level_id, config.DEFAULT_EPISODES)
                    EPSILON_DECAY = int(0.80 * EPISODES)
                    agent = QLearningAgent(EPSILON_DECAY, use_intrinsic) if algo_name == "Q_Learning" else SarsaAgent(EPSILON_DECAY, use_intrinsic)
                    rewards.clear()
                    intrinsic_rewards.clear()
                    episode_lengths.clear()
                    paused = True
                    training_done = False

                # Level selection
                for i, btn in enumerate(level_buttons):
                    if btn.clicked(pos) and level_id != i:
                        level_id = i
                        EPISODES = config.EPISODES_PER_LEVEL.get(level_id, config.DEFAULT_EPISODES)
                        EPSILON_DECAY = int(0.80 * EPISODES)
                        level_max_env_reward = compute_level_max_env_reward(LEVELS[level_id])

                        use_intrinsic = False
                        buttons["intrinsic"].toggle = False
                        
                        agent = QLearningAgent(EPSILON_DECAY, use_intrinsic) if algo_name == "Q_Learning" else SarsaAgent(EPSILON_DECAY, use_intrinsic)
                        env = GridWorld(LEVELS[level_id])
                        state = env.reset()
                        action = agent.select_action(state)

                        rewards.clear()
                        intrinsic_rewards.clear()
                        episode_lengths.clear()
                        episode_reward = 0
                        episode_intrinsic_reward = 0
                        steps = 0
                        paused = True
                        training_done = False

                # Play or Pause training
                if buttons["play"].clicked(pos):
                    paused = not paused
                    buttons["play"].toggle = not paused
                    buttons["play"].text = "Pause" if not paused else "Play"

                # Fast mode toggle
                if buttons["fast"].clicked(pos):
                    fast_mode = not fast_mode
                    buttons["fast"].toggle = fast_mode

                # Toggle intrinsic reward (only for Level 6)
                if buttons["intrinsic"].clicked(pos) and level_id == 6:
                    use_intrinsic = not use_intrinsic
                    buttons["intrinsic"].toggle = use_intrinsic
                    
                    # Create new agent with intrinsic reward setting
                    EPISODES = config.EPISODES_PER_LEVEL.get(level_id, config.DEFAULT_EPISODES)
                    EPSILON_DECAY = int(0.80 * EPISODES)
                    agent = QLearningAgent(EPSILON_DECAY, use_intrinsic) if algo_name == "Q_Learning" else SarsaAgent(EPSILON_DECAY, use_intrinsic)
                    
                    # Reset training
                    rewards.clear()
                    intrinsic_rewards.clear()
                    episode_lengths.clear()
                    episode_reward = 0
                    episode_intrinsic_reward = 0
                    steps = 0
                    paused = True
                    training_done = False
                    
                    env = GridWorld(LEVELS[level_id])
                    state = env.reset()
                    action = agent.select_action(state)

                # Save or Load model
                if buttons["save"].clicked(pos):
                    os.makedirs("models", exist_ok=True)
                    suffix = "_intrinsic" if use_intrinsic else ""
                    agent.save(f"models/{algo_name}_level{level_id}{suffix}.pkl")
                    save_training_curve(rewards, episode_lengths, algo_name, level_id, suffix)
                        
                if buttons["load"].clicked(pos):
                    suffix = "_intrinsic" if use_intrinsic else ""
                    path = f"models/{algo_name}_level{level_id}{suffix}.pkl"
                    if os.path.exists(path):
                        agent.load(path)

        # Training step
        if not paused and not training_done:
            next_state, env_reward, done = env.step(action)

            # Update agent based on selected algorithm
            if isinstance(agent, SarsaAgent):
                next_action = agent.select_action(next_state)
                intrinsic_reward = agent.update(state, action, env_reward, next_state, next_action)
                action = next_action
            else:
                intrinsic_reward = agent.update(state, action, env_reward, next_state)
                action = agent.select_action(next_state)

            state = next_state
            episode_reward += env_reward
            episode_intrinsic_reward += intrinsic_reward
            steps += 1

            # Episode termination
            if done or steps >= config.MAX_STEPS_PER_EPISODE:
                rewards.append(episode_reward)
                intrinsic_rewards.append(episode_intrinsic_reward)
                episode_lengths.append(steps)
                agent.new_episode()

                # Stop training and save training curve/model after configured number of episodes
                if agent.episode >= EPISODES:
                    training_done = True
                    paused = True
                    suffix = "_intrinsic" if use_intrinsic else ""
                    save_training_curve(rewards, episode_lengths, algo_name, level_id, suffix)

                    # Auto-save trained model
                    os.makedirs("models", exist_ok=True)
                    model_path = f"models/{algo_name}_level{level_id}{suffix}.pkl"
                    agent.save(model_path)
                    print(f"[Saved model] {model_path}")

                    # Print stats
                    print(f"\n{'='*60}")
                    print(f"Training Complete - Level {level_id}")
                    print(f"Algorithm: {algo_name}")
                    print(f"Intrinsic Reward: {'YES' if use_intrinsic else 'NO'}")
                    print(f"Episodes: {agent.episode}/{EPISODES}")
                    if len(rewards) >= 50:
                        final_avg = sum(rewards[-50:]) / 50
                        print(f"Avg reward (last 50): {final_avg:.2f}")
                    if use_intrinsic and len(intrinsic_rewards) >= 50:
                        final_intrinsic = sum(intrinsic_rewards[-50:]) / 50
                        print(f"Avg intrinsic (last 50): {final_intrinsic:.3f}")
                    print(f"{'='*60}\n")

                env = GridWorld(LEVELS[level_id])
                state = env.reset()
                action = agent.select_action(state)

                episode_reward = 0
                episode_intrinsic_reward = 0
                steps = 0

        # --------------------------------------------------
        # Rendering
        # --------------------------------------------------
        game_surface.fill((0, 0, 0))
        render(game_surface, env, tiles, monsters, agent_sprite)
        screen.blit(game_surface, (0, 0))

        pygame.draw.rect(
            screen, (30, 30, 30),
            (TILE_W + 10, 380, PANEL_W - 20, 265),
            border_radius=8
        )
        
        window = 50
        px = TILE_W + 40
        y = 400
        line = 26

        # Display stats
        draw_text(screen, f"Algorithm: {algo_name.upper()}", px, y); y += line
        draw_text(screen, f"Level: {level_id}", px, y); y += line
        draw_text(screen, f"Episode: {agent.episode}/{EPISODES}", px, y); y += line
        draw_text(screen, f"Epsilon: {agent.epsilon():.3f}", px, y); y += line
        
        # Show intrinsic reward status
        if level_id == 6: 
            intrinsic_status = "ON" if use_intrinsic else "OFF"
            color = (100, 255, 100) if use_intrinsic else (200, 200, 200)
            draw_text(screen, f"Intrinsic: {intrinsic_status}", px, y, color=color); y += line
        
        # Current episode rewards
        draw_text(screen, f"Env Reward: {episode_reward} / {level_max_env_reward}", px, y,); y += line
        
        if use_intrinsic and episode_intrinsic_reward > 0:
            draw_text(screen, f"Intrinsic: +{episode_intrinsic_reward:.2f}", px, y, color=(150, 150, 255)); y += line

        # Rolling average rewards over the last window episodes
        if len(rewards) >= window:
            avg_env = sum(rewards[-window:]) / window
            draw_text(screen, f"Avg Reward: {avg_env:.2f} / {level_max_env_reward:.2f}", px, y)
            y += line

        if len(episode_lengths) >= window:
            avg_steps = sum(episode_lengths[-window:]) / window
            draw_text(screen, f"Avg Steps: {avg_steps:.1f}", px, y)
            y += line

        if training_done:
            draw_text(screen, "Training complete", px, y, size=20, color=(0, 255, 0))

        # Button states
        buttons["q"].active = (algo_name == "Q_Learning")
        buttons["s"].active = (algo_name == "SARSA")
        buttons["intrinsic"].enabled = (level_id == 6) 

        for i, btn in enumerate(level_buttons):
            btn.active = (i == level_id)

        for b in buttons.values():
            b.draw(screen, font)
        for b in level_buttons:
            b.draw(screen, font)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()