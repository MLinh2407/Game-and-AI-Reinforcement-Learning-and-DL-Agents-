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

# Save a training curve showing episode rewards over time
def save_training_curve(rewards, algo, level_id):
    if len(rewards) < 2:
        return
    
    window = 50
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(rewards, label="Episode reward", alpha=0.7, linewidth=1)

    if len(rewards) >= window:
        smooth = [
            sum(rewards[max(0, i - window):i + 1]) /
            (i - max(0, i - window) + 1)
            for i in range(len(rewards))
        ]
        plt.plot(smooth, label=f"Average reward", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{algo} – Level {level_id}")
    plt.legend()
    plt.tight_layout()

    path = f"plots/{algo}_level{level_id}_training.png"
    plt.savefig(path)
    plt.close()

    print(f"[Saved training curve] {path}")

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
    
    EPISODES = config.EPISODES_PER_LEVEL.get(level_id, config.DEFAULT_EPISODES)
    EPSILON_DECAY = int(0.85 * EPISODES)
    
    agent = QLearningAgent(EPSILON_DECAY)

    # Initialize environment and agent state
    env = GridWorld(LEVELS[level_id])
    state = env.reset()
    action = agent.select_action(state)

    paused = True
    fast_mode = False
    training_done = False

    episode_reward = 0
    rewards = []
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
                save_training_curve(rewards, algo_name, level_id)
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
                    EPSILON_DECAY = int(0.85 * EPISODES)
                    agent = QLearningAgent(EPSILON_DECAY) if algo_name == "Q_Learning" else SarsaAgent(EPSILON_DECAY)
                    rewards.clear()
                    episode_lengths.clear()
                    paused = True
                    training_done = False

                # Level selection
                for i, btn in enumerate(level_buttons):
                    if btn.clicked(pos) and level_id != i:
                        level_id = i
                        EPISODES = config.EPISODES_PER_LEVEL.get(level_id, config.DEFAULT_EPISODES)
                        EPSILON_DECAY = int(0.85 * EPISODES)

                        agent = QLearningAgent(EPSILON_DECAY) if algo_name == "Q_Learning" else SarsaAgent(EPSILON_DECAY)
                        env = GridWorld(LEVELS[level_id])
                        state = env.reset()
                        action = agent.select_action(state)

                        rewards.clear()
                        episode_lengths.clear()
                        episode_reward = 0
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

                # Save or Load model
                if buttons["save"].clicked(pos):
                    os.makedirs("models", exist_ok=True)
                    agent.save(f"models/{algo_name}_level{level_id}.pkl")
                    save_training_curve(rewards, algo_name, level_id)

                if buttons["load"].clicked(pos):
                    path = f"models/{algo_name}_level{level_id}.pkl"
                    if os.path.exists(path):
                        agent.load(path)

        # Training step
        if not paused and not training_done:
            next_state, reward, done = env.step(action)

            # Update agent based on selected algorithm
            if isinstance(agent, SarsaAgent):
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state)
                action = agent.select_action(next_state)

            state = next_state
            episode_reward += reward
            steps += 1

            # Episode termination
            if done or steps >= config.MAX_STEPS_PER_EPISODE:
                rewards.append(episode_reward)
                episode_lengths.append(steps)
                agent.new_episode()

                # Stop training and save training curve after configured number of episodes
                if agent.episode >= EPISODES:
                    training_done = True
                    paused = True
                    save_training_curve(rewards, algo_name, level_id)

                env = GridWorld(LEVELS[level_id])
                state = env.reset()
                action = agent.select_action(state)

                episode_reward = 0
                steps = 0

        # --------------------------------------------------
        # Rendering
        # --------------------------------------------------
        game_surface.fill((0, 0, 0))
        render(game_surface, env, tiles, monsters, agent_sprite)
        screen.blit(game_surface, (0, 0))

        pygame.draw.rect(
            screen, (30, 30, 30),
            (TILE_W + 10, 400, PANEL_W - 20, 240),
            border_radius=8
        )
        
        window = 50
        px = TILE_W + 40
        y = 420
        line = 26

        # Display training statistics
        draw_text(screen, f"Algorithm: {algo_name.upper()}", px, y); y += line
        draw_text(screen, f"Level: {level_id}", px, y); y += line
        draw_text(screen, f"Episode: {agent.episode}/{EPISODES}", px, y); y += line
        draw_text(screen, f"Epsilon: {agent.epsilon():.3f}", px, y); y += line
        draw_text(screen, f"Episode Reward: {episode_reward}", px, y); y += line

        if len(rewards) >= window:
            avg = sum(rewards[-window:]) / window
            draw_text(screen, f"Avg Reward ({window}): {avg:.2f}", px, y)
            y += line

        if len(episode_lengths) >= window:
            avg_steps = sum(episode_lengths[-window:]) / window
            draw_text(screen, f"Avg Steps ({window}): {avg_steps:.1f}", px, y)

        if training_done:
            draw_text(screen, "Training complete", px, y + 30, size=22, color=(0, 255, 0))

        buttons["q"].active = (algo_name == "Q_Learning")
        buttons["s"].active = (algo_name == "SARSA")

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