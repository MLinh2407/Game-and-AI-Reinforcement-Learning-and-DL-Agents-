import pygame
import random
import sys

from constants import *
from levels import LEVELS
from gridworld import GridWorld
from assets import load_tiles
from rendering import render

def main():
    pygame.init()

    # -------------------------
    # Select level
    # -------------------------
    level_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = GridWorld(LEVELS[level_id])

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
    # Game surface
    # -------------------------
    GAME_WIDTH  = len(env.grid[0]) * TILE_SIZE
    GAME_HEIGHT = len(env.grid) * TILE_SIZE

    game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))

    clock = pygame.time.Clock()
    running = True

    # -------------------------
    # Main loop
    # -------------------------
    while running:
        # turn-based speed
        clock.tick(8)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
