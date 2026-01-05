from constants import *
import random

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

        # Assign a random seed to each monster tile
        self.monster_seeds = {}
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile == MONSTER:
                    self.monster_seeds[(x, y)] = random.random()

        return self.get_state()

    def get_state(self):
        return tuple(self.agent_pos)

    # Apply an action and update the environment
    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        dx, dy = ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = 0

        # Check grid boundaries
        if 0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0]):
            tile = self.grid[ny][nx]

            # Rock tiles block movement
            if tile != ROCK:
                self.agent_pos = [nx, ny]

                # Apple gives reward
                if tile == APPLE:
                    reward = 1
                    self.grid[ny][nx] = FLOOR

                # Fire and monster end the episode
                elif tile == FIRE:
                    self.done = True
                    
                elif tile == MONSTER:
                    self.done = True
                    
                # Key enables chest reward
                elif tile == KEY:
                    self.has_key = True
                    self.grid[ny][nx] = FLOOR

                elif tile == CHEST and self.has_key:
                    reward = 2
                    self.grid[ny][nx] = FLOOR

        #After agent action, monsters may move
        if not self.done:
            self.update_monsters()
            
        # End episode if all collectibles are gone
        if self.all_collected():
            self.done = True

        return self.get_state(), reward, self.done

    # Monsters has a 40% chance to move after agent action
    def update_monsters(self):
        monsters = {}
        for (x,y), seed in self.monster_seeds.items():
            #40% chance monster attempt to move
            if random.random() < 0.4:
            
                # Try random directions
                directions = list(ACTIONS.values())
                random.shuffle(directions)
                
                moved = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0]):
                        # Monster hit agent, episode ends
                        if [nx,ny] == self.agent_pos:
                            self.done = True
                            self.grid[y][x] = FLOOR
                            self.grid[ny][nx] = MONSTER
                            monsters[(nx,ny)] = seed
                            moved = True
                            break
                        
                        # Monster moves to empty floor
                        if self.grid[ny][nx] == FLOOR:
                            self.grid[y][x] = FLOOR
                            self.grid[ny][nx] = MONSTER
                            monsters[(nx,ny)] = seed
                            moved = True
                            break
            
                # Monster failed to move            
                if not moved:
                    monsters[(x,y)] = seed
        
            else:
                # 60% chance monster stays in place
                monsters[(x,y)] = seed
                
        self.monster_seeds = monsters
        
    # Check if all apples and chests are collected
    def all_collected(self):
        for row in self.grid:
            for tile in row:
                if tile in (APPLE, CHEST):
                    return False
        return True
