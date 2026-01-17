import pygame
import os
import config

def load_scaled_image(path, size, *, alpha):
    if not os.path.exists(path):
        return None
    image = pygame.image.load(path)
    image = image.convert_alpha() if alpha else image.convert()
    return pygame.transform.scale(image, size)

def load_assets():
    if not os.path.exists(config.ASSETS_PATH):
        os.makedirs(config.ASSETS_PATH)

    # Load background
    bg_path = os.path.join(config.ASSETS_PATH, config.BACKGROUND_IMAGE)
    background = load_scaled_image(
        bg_path,
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
        alpha=False,
    )

    # Load player sprite
    player_path = os.path.join(config.ASSETS_PATH, config.PLAYER_SHIP_IMAGE)
    player_sprite = load_scaled_image(
        player_path,
        (config.PLAYER_SIZE, config.PLAYER_SIZE),
        alpha=True,
    )

    # Load enemy sprite
    enemy_path = os.path.join(config.ASSETS_PATH, config.ENEMY_SHIP_IMAGE)
    enemy_sprite = load_scaled_image(
        enemy_path,
        (config.ENEMY_SIZE, config.ENEMY_SIZE),
        alpha=True,
    )

    # Load spawner sprite
    spawner_path = os.path.join(config.ASSETS_PATH, config.SPAWNER_IMAGE)
    spawner_sprite = load_scaled_image(
        spawner_path,
        (config.SPAWNER_SIZE, config.SPAWNER_SIZE),
        alpha=True,
    )

    # Load bullet sprite
    bullet_path = os.path.join(config.ASSETS_PATH, config.BULLET_IMAGE)
    bullet_sprite = load_scaled_image(
        bullet_path,
        (config.BULLET_SIZE, config.BULLET_SIZE * 2),
        alpha=True,
    )

    return background, player_sprite, enemy_sprite, spawner_sprite, bullet_sprite