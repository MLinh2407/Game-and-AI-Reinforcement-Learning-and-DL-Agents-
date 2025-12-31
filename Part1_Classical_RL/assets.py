import pygame
import os
from constants import *

ASSET_PATH = "assets/"

# Mapping from tile ID to image file
def load_tiles():
    tiles = {
        FLOOR:  "floor.png",
        ROCK:   "rock.png",
        APPLE:  "apple.png",
        FIRE:   "fire.png",
        KEY:    "key.png",
        CHEST:  "chest.png",
    }

    # Load and scale tile images
    images = {}
    for tile_id, filename in tiles.items():
        img = pygame.image.load(ASSET_PATH + filename).convert_alpha()
        images[tile_id] = pygame.transform.scale(
            img, (TILE_SIZE, TILE_SIZE)
        )

    # Monster sprites
    monsters = []
    for filename in os.listdir(ASSET_PATH):
        if filename.startswith("monster") and filename.endswith(".png"):
            img = pygame.image.load(
                ASSET_PATH + filename
            ).convert_alpha()

            monsters.append(
                pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
            )

    # Agent sprite
    agent = pygame.image.load(
        ASSET_PATH + "agent.png"
    ).convert_alpha()
    agent = pygame.transform.scale(agent, (TILE_SIZE, TILE_SIZE))

    return images, monsters, agent
