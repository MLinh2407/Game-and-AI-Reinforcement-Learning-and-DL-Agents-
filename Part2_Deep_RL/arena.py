import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config
from player import Player
from enemy import Enemy
from spawner import Spawner
from projectile import Projectile
from particles import ParticleSystem
from rendering import Renderer

class ArenaEnvironment(gym.Env):
    def __init__(self, control_scheme='rotation', render_mode=None):
        super(ArenaEnvironment, self).__init__()
        
        # Window settings
        self.width = config.WINDOW_WIDTH
        self.height = config.WINDOW_HEIGHT
        self.render_mode = render_mode
        self.control_scheme = control_scheme
        
        # Define action space
        if control_scheme == config.CONTROL_ROTATION:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_ROTATION)
        else:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_DIRECTIONAL)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.OBSERVATION_SIZE,), 
            dtype=np.float32
        )
        
        # Initialize pygame
        pygame.init()
        
        # Initialize systems
        self.player = Player(self.width // 2, self.height // 2, control_scheme)
        self.enemies = []
        self.spawners = []
        self.bullets = []
        self.particle_system = ParticleSystem()
        self.renderer = Renderer(self.width, self.height)
        
        # Game state
        self.current_phase = config.PHASE_START
        self.step_count = 0
        self.max_steps = config.MAX_STEPS
        
        # Stats
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0
    
    # Reset the environment to initial state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset player
        self.player.reset(self.width // 2, self.height // 2)
        
        # Clear all entities
        self.enemies = []
        self.spawners = []
        self.bullets = []
        self.particle_system.clear()
        
        # Reset game state
        self.current_phase = config.PHASE_START
        self.step_count = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0
        
        # Spawn initial spawners
        self.spawn_phase_spawners()
        
        return self.get_observation(), {}
    
    def step(self, action):
        self.step_count += 1
        reward = 0.0
        
        # Process player action
        reward += self.process_player_action(action)
        
        # Update all entities
        self.player.update((self.width, self.height))
        self.update_spawners()
        self.update_enemies()
        self.update_bullets()
        self.particle_system.update()
        
        # Check collisions
        reward += self.check_collisions()
        
        # Check phase completion
        if len(self.spawners) == 0:
            reward += config.REWARD_PHASE_COMPLETE
            self.current_phase += 1
            self.spawn_phase_spawners()
            self.particle_system.phase_complete_effect(self.player.pos)
        
        # Check termination
        done = False
        if not self.player.is_alive():
            reward += config.REWARD_DEATH
            done = True
            self.particle_system.player_explosion(self.player.pos)
        
        if self.step_count >= self.max_steps:
            done = True
        
        # Survival reward
        reward += config.REWARD_SURVIVAL
        
        observation = self.get_observation()
        info = {
            'phase': self.current_phase,
            'enemies_destroyed': self.enemies_destroyed,
            'spawners_destroyed': self.spawners_destroyed,
            'player_health': self.player.health
        }
        
        return observation, reward, done, False, info
    
    # Spawn spawners based on current phase
    def spawn_phase_spawners(self):
        self.spawners = []
        
        num_spawners = min(
            config.SPAWNER_MIN_COUNT + self.current_phase,
            config.SPAWNER_MAX_COUNT
        )
        
        # Place spawners around center
        for i in range(num_spawners):
            angle = (2 * np.pi * i) / num_spawners
            distance = min(self.width, self.height) * 0.35
            x = self.width // 2 + distance * np.cos(angle)
            y = self.height // 2 + distance * np.sin(angle)
            
            spawner = Spawner(x, y, self.current_phase)
            self.spawners.append(spawner)
    
    # Process player action based on control scheme
    def process_player_action(self, action):
        reward = 0.0
        
        if self.control_scheme == config.CONTROL_ROTATION:
            # Action: 0=no-op, 1=thrust, 2=rotate_left, 3=rotate_right, 4=shoot
            if action == 1:
                self.player.move_thrust()
                thrust_pos = self.player.get_thrust_particle_pos()
                self.particle_system.thrust_particle(thrust_pos, self.player.vel)
            elif action == 2:
                self.player.rotate('left')
            elif action == 3:
                self.player.rotate('right')
            elif action == 4:
                reward += self.player_shoot()
        else:
            # Action: 0=no-op, 1=up, 2=down, 3=left, 4=right, 5=shoot
            if action == 1:
                self.player.move_direction('up')
            elif action == 2:
                self.player.move_direction('down')
            elif action == 3:
                self.player.move_direction('left')
            elif action == 4:
                self.player.move_direction('right')
            elif action == 5:
                reward += self.player_shoot()
        
        return reward
    
    # Handle player shooting
    def player_shoot(self):
        if self.player.can_shoot(self.step_count):
            # Get nearest enemy for directional control
            nearest_enemy = self.nearest_enemy()
            target_pos = nearest_enemy.pos if nearest_enemy else None
            
            vel, angle = self.player.shoot_velocity(target_pos)
            bullet = Projectile(
                self.player.pos[0], self.player.pos[1],
                vel[0], vel[1], angle
            )
            self.bullets.append(bullet)
            self.player.last_shot_time = self.step_count
        
        return 0.0
    
    def update_spawners(self):
        for spawner in self.spawners:
            spawner.update()
            if spawner.should_spawn():
                self.spawn_enemy(spawner.pos)
    
    def spawn_enemy(self, pos):
        # Add some randomness to spawn position
        offset = np.random.randn(2) * 20
        enemy = Enemy(pos[0] + offset[0], pos[1] + offset[1], self.current_phase)
        self.enemies.append(enemy)
        self.particle_system.spawn_effect(enemy.pos)
    
    def update_enemies(self):
        for enemy in self.enemies:
            enemy.update(self.player.pos)
    
    # Update all bullets and remove out of bounds
    def update_bullets(self):
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            bullet.update()
            if bullet.is_out_of_bounds(self.width, self.height):
                bullets_to_remove.append(i)
        
        # Remove bullets in reverse order
        for i in sorted(bullets_to_remove, reverse=True):
            del self.bullets[i]
    
    # Check all collisions and return reward
    def check_collisions(self):
        reward = 0.0
        
        bullets_to_remove = []
        enemies_to_remove = []
        spawners_to_remove = []
        
        # Bullets vs Enemies
        for i, bullet in enumerate(self.bullets):
            if i in bullets_to_remove:
                continue
            for j, enemy in enumerate(self.enemies):
                if j in enemies_to_remove:
                    continue
                if bullet.collides_with(enemy.pos, config.ENEMY_COLLISION_RADIUS):
                    if not enemy.take_damage(bullet.damage):
                        enemies_to_remove.append(j)
                        reward += config.REWARD_ENEMY_KILL
                        self.enemies_destroyed += 1
                        self.particle_system.enemy_explosion(enemy.pos)
                    else:
                        self.particle_system.hit_effect(
                            enemy.pos, config.COLOR_EXPLOSION_ENEMY)
                    bullets_to_remove.append(i)
                    break
        
        # Bullets vs Spawners
        for i, bullet in enumerate(self.bullets):
            if i in bullets_to_remove:
                continue
            for j, spawner in enumerate(self.spawners):
                if j in spawners_to_remove:
                    continue
                if bullet.collides_with(spawner.pos, config.SPAWNER_COLLISION_RADIUS):
                    if not spawner.take_damage(bullet.damage):
                        spawners_to_remove.append(j)
                        reward += config.REWARD_SPAWNER_DESTROY
                        self.spawners_destroyed += 1
                        self.particle_system.spawner_explosion(spawner.pos)
                    else:
                        self.particle_system.hit_effect(
                            spawner.pos, config.COLOR_EXPLOSION_SPAWNER)
                    bullets_to_remove.append(i)
                    break
        
        # Remove destroyed entities
        for i in sorted(set(bullets_to_remove), reverse=True):
            del self.bullets[i]
        for i in sorted(set(enemies_to_remove), reverse=True):
            del self.enemies[i]
        for i in sorted(set(spawners_to_remove), reverse=True):
            del self.spawners[i]
        
        # Player vs Enemies collision
        for enemy in self.enemies:
            if enemy.collides_with(self.player.pos, config.PLAYER_COLLISION_RADIUS):
                self.player.take_damage(config.ENEMY_COLLISION_DAMAGE)
                reward += config.REWARD_DAMAGE_TAKEN
                self.particle_system.hit_effect(
                    self.player.pos, config.COLOR_EXPLOSION_PLAYER)
        
        return reward
    
    # Get nearest enemy to player
    def nearest_enemy(self):
        if len(self.enemies) == 0:
            return None
        return min(self.enemies, 
                  key=lambda e: np.linalg.norm(e.pos - self.player.pos))
    
    # Get nearest spawner to player
    def nearest_spawner(self):
        if len(self.spawners) == 0:
            return None
        return min(self.spawners, 
                  key=lambda s: np.linalg.norm(s.pos - self.player.pos))
    
    # Get current observation vector
    def get_observation(self):
        obs = np.zeros(config.OBSERVATION_SIZE, dtype=np.float32)
        
        # Player state (normalized)
        obs[0] = self.player.pos[0] / self.width
        obs[1] = self.player.pos[1] / self.height
        obs[2] = self.player.vel[0] / config.PLAYER_MAX_SPEED
        obs[3] = self.player.vel[1] / config.PLAYER_MAX_SPEED
        obs[4] = self.player.angle / (2 * np.pi)

        # Nearest enemy
        nearest_enemy = self.nearest_enemy()
        if nearest_enemy is not None:
            diff = nearest_enemy.pos - self.player.pos
            distance = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            obs[5] = distance / np.sqrt(self.width**2 + self.height**2)
            obs[6] = angle / (2 * np.pi)
        else:
            obs[5] = 1.0
            obs[6] = 0.0

        # Nearest spawner
        nearest_spawner = self.nearest_spawner()
        if nearest_spawner is not None:
            diff = nearest_spawner.pos - self.player.pos
            distance = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            obs[7] = distance / np.sqrt(self.width**2 + self.height**2)
            obs[8] = angle / (2 * np.pi)
        else:
            obs[7] = 1.0
            obs[8] = 0.0

        # Health and phase
        obs[9] = self.player.health / self.player.max_health
        obs[10] = self.current_phase / 10.0

        return obs
    
    # Render the environment
    def render(self):
        self.renderer.initialize()

        self.renderer.draw_background()
        self.renderer.draw_particles(self.particle_system.get_particles())
        
        for spawner in self.spawners:
            self.renderer.draw_spawner(spawner)
        
        for enemy in self.enemies:
            self.renderer.draw_enemy(enemy)
        
        for bullet in self.bullets:
            self.renderer.draw_bullet(bullet)
        
        self.renderer.draw_player(self.player)
        
        self.renderer.draw_ui(
            self.player.health,
            self.current_phase,
            len(self.enemies),
            len(self.spawners)
        )
        
        self.renderer.update_display()
    
    def close(self):
        self.renderer.close()