from constants import *

def render(screen, env, tiles, monsters, agent):
    monster_count = len(monsters)

    for y, row in enumerate(env.grid):
        for x, tile in enumerate(row):
            pos = (x * TILE_SIZE, y * TILE_SIZE)

            # Draw base floor tile
            screen.blit(tiles[FLOOR], pos)

            if tile == MONSTER:
                # Stable sprite selection using seed
                seed = env.monster_seeds[(x, y)]
                idx = int(seed * monster_count)
                screen.blit(monsters[idx], pos)

            # Draw other objects (rock, apple, fire, key, chest)
            elif tile != FLOOR:
                screen.blit(tiles[tile], pos)

    # Draw agent
    ax, ay = env.agent_pos
    screen.blit(agent, (ax * TILE_SIZE, ay * TILE_SIZE))
