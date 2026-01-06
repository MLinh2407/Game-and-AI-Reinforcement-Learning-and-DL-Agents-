from constants import *
import random
import config

class GridWorld:
    def __init__(self, grid):
        self.original_grid = grid
        self.reset()

    def reset(self):
        # Reset grid and agent state
        self.grid = [row[:] for row in self.original_grid]
        self.agent_pos = [0, 0]
        self.has_key = False
        self.done = False

        # Assign a stable random seed to each monster
        self.monster_seeds = {}
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile == MONSTER:
                    self.monster_seeds[(x, y)] = random.random()

        return self.get_state()

    # State representation
    def get_state(self):
        """
        State includes:
        - Agent x position
        - Agent y position
        - Whether the agent has the key (0 or 1)
        """
        return (self.agent_pos[0], self.agent_pos[1], int(self.has_key))

    # Environment step
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        dx, dy = ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = 0

        # Check boundaries
        if 0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0]):
            tile = self.grid[ny][nx]

            # Rocks block movement
            if tile != ROCK:
                self.agent_pos = [nx, ny]

                # Apple 
                if tile == APPLE:
                    reward = 1
                    self.grid[ny][nx] = FLOOR

                # Fire = death penalty
                elif tile == FIRE:
                    reward = config.DEATH_PENALTY
                    self.done = True

                # Monster = death penalty
                elif tile == MONSTER:
                    reward = config.DEATH_PENALTY
                    self.done = True

                # Key
                elif tile == KEY:
                    self.has_key = True
                    self.grid[ny][nx] = FLOOR

                # Chest (only works if key collected)
                elif tile == CHEST and self.has_key:
                    reward = 2
                    self.grid[ny][nx] = FLOOR

        # Monsters move after agent action
        if not self.done:
            self.update_monsters()

        # If monster moved into agent, apply death penalty
        if self.done:
            reward = config.DEATH_PENALTY

        # End episode if all collectibles obtained
        if self.all_collected():
            self.done = True

        return self.get_state(), reward, self.done

    # Monster movement (stochastic)
    def update_monsters(self):
        new_monsters = {}

        for (x, y), seed in self.monster_seeds.items():
            # 40% chance monster attempts to move
            if random.random() < 0.4:
                directions = list(ACTIONS.values())
                random.shuffle(directions)

                moved = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0]):
                        # Monster hits agent
                        if [nx, ny] == self.agent_pos:
                            self.done = True
                            self.grid[y][x] = FLOOR
                            self.grid[ny][nx] = MONSTER
                            new_monsters[(nx, ny)] = seed
                            moved = True
                            break

                        # Monster moves to empty floor
                        if self.grid[ny][nx] == FLOOR:
                            self.grid[y][x] = FLOOR
                            self.grid[ny][nx] = MONSTER
                            new_monsters[(nx, ny)] = seed
                            moved = True
                            break

                if not moved:
                    new_monsters[(x, y)] = seed
            else:
                # Monster stays in place
                new_monsters[(x, y)] = seed

        self.monster_seeds = new_monsters

    # Termination condition
    def all_collected(self):
        for row in self.grid:
            for tile in row:
                if tile in (APPLE, CHEST):
                    return False
        return True