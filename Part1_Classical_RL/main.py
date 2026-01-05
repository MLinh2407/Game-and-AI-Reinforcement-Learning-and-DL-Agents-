import pygame
import random
import sys

from constants import *
from levels import LEVELS
from gridworld import GridWorld
from assets import load_tiles
from rendering import render
from sarsa import epsilon_for_episode, epsilon_greedy_action, get_q_list, greedy_action
from config import FPS_VISUAL, FPS_FAST, EPISODES, MAX_STEPS_PER_EPISODE, ALPHA, GAMMA

def main():
    pygame.init()

    # -------------------------
    # Select level
    # -------------------------
    level_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0


    # -------------------------
    # Create window
    # -------------------------
    WINDOW_WIDTH  = 670
    WINDOW_HEIGHT = 670

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Part I – Gridworld (Level {level_id})")

    # -------------------------
    # Load assets
    # -------------------------
    tiles, monsters, agent = load_tiles()

    # -------------------------
    # Environment and game surface
    # -------------------------
    env = GridWorld(LEVELS[level_id])

    GAME_WIDTH  = len(env.grid[0]) * TILE_SIZE
    GAME_HEIGHT = len(env.grid) * TILE_SIZE

    game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))

    # -------------------------
    # If level 1 or 3, run SARSA training with HUD in main (V/R controls)
    # -------------------------
    visualize_policy = False
    if level_id == 1 or level_id == 3:
        Q = {}
        fast_mode = False
        training_running = True
        font = pygame.font.SysFont("consolas", 18)

        for ep in range(EPISODES):
            state = env.reset()
            epsilon = epsilon_for_episode(ep)
            action = epsilon_greedy_action(Q, state, epsilon)
            total_reward = 0.0

            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        training_running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_v:
                            fast_mode = not fast_mode
                        if event.key == pygame.K_r:
                            Q = {}
                            print("Q-table reset by user")
                if not training_running:
                    break

                # SARSA step/update
                next_state, reward, done = env.step(action)
                next_action = epsilon_greedy_action(Q, next_state, epsilon)

                q_s = get_q_list(Q, state)
                q_next = get_q_list(Q, next_state)

                td_target = reward + (0 if done else GAMMA * q_next[next_action])
                td_error = td_target - q_s[action]
                q_s[action] += ALPHA * td_error

                state, action = next_state, next_action
                total_reward += reward

                # Render environment and HUD
                render(game_surface, env, tiles, monsters, agent)
                # scale and blit
                scaled = pygame.transform.scale(game_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
                screen.blit(scaled, (0, 0))
                # HUD overlay (top-left)
                hud = [
                    f"Ep {ep+1}/{EPISODES}  step {step}  eps {epsilon:.3f}",
                    f"Rewards left {env.rewards_left()}",
                    f"Return {total_reward:.2f}  (SARSA training)",
                    "V to toggles fast mode. R to resets. Close window to stop training."
                ]
                for i, line in enumerate(hud):
                    screen.blit(font.render(line, True, (240,240,240)), (8, 8 + i*18))

                pygame.display.flip()
                # pacing
                pygame.time.Clock().tick(FPS_FAST if fast_mode else FPS_VISUAL)

                if done:
                    break

            print(f"Episode {ep+1}/{EPISODES} finished in {step} steps, return {total_reward:.2f}, eps {epsilon:.3f}")

            if not training_running:
                print("Training stopped by user.")
                break

        print("SARSA training completed")
        print("Visualizing learned policy (greedy, no exploration)...")
        env = GridWorld(LEVELS[level_id])
        visualize_policy = True

    clock = pygame.time.Clock()
    running = True

    # -------------------------
    # Main loop
    # -------------------------
    while running:
        # Use visual FPS if showing learned policy, otherwise keep turn-based speed
        fps = FPS_VISUAL if visualize_policy else 8
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # If we are visualizing the learned SARSA policy, step the environment greedily
        if visualize_policy and not env.done:
            state = env.get_state()
            action = greedy_action(Q, state)
            env.step(action)

        # Render environment
        render(game_surface, env, tiles, monsters, agent)

        scaled_surface = pygame.transform.scale(
            game_surface,
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
