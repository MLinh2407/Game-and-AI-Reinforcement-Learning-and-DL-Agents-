import pygame
import numpy as np
import config
from assets import load_assets

class Renderer:
    # Return a color based on remaining health ratio
    @staticmethod
    def health_color_from_ratio(health_ratio, *, high, mid):
        if health_ratio > high:
            return (0, 255, 0)
        if health_ratio > mid:
            return (255, 255, 0)
        return (255, 0, 0)

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = None
        
        # Sprites (loaded once)
        self.background = None
        self.player_sprite = None
        self.enemy_sprite = None
        self.spawner_sprite = None
        self.bullet_sprite = None
        self.assets_loaded = False
        # UI buttons (created on initialize)
        self.buttons = None
    
    # Initialize pygame window and load assets
    def initialize(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Shooting Arena - Deep RL")
            self.font = pygame.font.Font(None, 24)
            
            # Load all assets
            if not self.assets_loaded:
                (self.background, self.player_sprite, self.enemy_sprite, 
                 self.spawner_sprite, self.bullet_sprite) = load_assets()
                self.assets_loaded = True

            # Create UI buttons (right side)
            if self.buttons is None:
                try:
                    from ui import create_ui
                    self.buttons = create_ui(self.width)
                except Exception:
                    self.buttons = None
    
    def draw_background(self):
        if self.background is not None:
            self.screen.blit(self.background, (0, 0))
        else:
            # Fallback
            self.screen.fill(config.COLOR_BG)
            for x in range(0, self.width, 40):
                pygame.draw.line(self.screen, config.COLOR_GRID, 
                               (x, 0), (x, self.height), 1)
            for y in range(0, self.height, 40):
                pygame.draw.line(self.screen, config.COLOR_GRID, 
                               (0, y), (self.width, y), 1)
    
    def draw_particles(self, particles):
        for particle in particles:
            color = particle.color
            size = particle.get_size()
            pos = particle.pos.astype(int)
            pygame.draw.circle(self.screen, color, pos, size)
    
    def draw_player(self, player):
        pos = player.pos.astype(int)
        
        if self.player_sprite is not None:
            # Rotate sprite to face direction
            rotated = pygame.transform.rotate(
                self.player_sprite, 
                -np.degrees(player.angle) - 90
            )
            rect = rotated.get_rect(center=pos)
            self.screen.blit(rotated, rect)
        else:
            # Fallback
            if player.control_scheme == config.CONTROL_ROTATION:
                points = [
                    player.pos + 20 * np.array([np.cos(player.angle), np.sin(player.angle)]),
                    player.pos + 15 * np.array([np.cos(player.angle + 2.5), np.sin(player.angle + 2.5)]),
                    player.pos + 15 * np.array([np.cos(player.angle - 2.5), np.sin(player.angle - 2.5)])
                ]
                pygame.draw.polygon(self.screen, config.COLOR_PLAYER, points, 0)
                pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
            else:
                pygame.draw.circle(self.screen, config.COLOR_PLAYER, pos, 15)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 15, 2)
    
    def draw_enemy(self, enemy):
        pos = enemy.pos.astype(int)
        
        if self.enemy_sprite is not None:
            rotated = pygame.transform.rotate(self.enemy_sprite, -np.degrees(enemy.angle) - 90)
            rect = rotated.get_rect(center=pos)
            self.screen.blit(rotated, rect)
        else:
            points = [
                enemy.pos + 15 * np.array([np.cos(enemy.angle), np.sin(enemy.angle)]),
                enemy.pos + 10 * np.array([np.cos(enemy.angle + 2.5), np.sin(enemy.angle + 2.5)]),
                enemy.pos + 10 * np.array([np.cos(enemy.angle - 2.5), np.sin(enemy.angle - 2.5)])
            ]
            points = [(int(p[0]), int(p[1])) for p in points]
            
            pygame.draw.polygon(self.screen, config.COLOR_ENEMY, points, 0)
            pygame.draw.polygon(self.screen, (255, 100, 100), points, 2)
        
        # Health bar
        health_ratio = enemy.health_ratio()
        bar_width = 30
        bar_height = 4
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] - 25
        
        # Background
        pygame.draw.rect(self.screen, (100, 0, 0),
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Health (color based on health ratio)
        health_color = self.health_color_from_ratio(health_ratio, high=0.5, mid=0.25)
        
        pygame.draw.rect(self.screen, health_color,
                        (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
    
    def draw_spawner(self, spawner):
        pos = spawner.pos.astype(int)
        
        if self.spawner_sprite is not None:
            rect = self.spawner_sprite.get_rect(center=pos)
            self.screen.blit(self.spawner_sprite, rect)
        else:
            # Fallback
            pygame.draw.circle(self.screen, config.COLOR_SPAWNER, pos, 25, 3)
            pygame.draw.circle(self.screen, config.COLOR_SPAWNER, pos, 15, 2)
        
        # Health bar
        health_ratio = spawner.health_ratio()
        bar_width = 40
        bar_height = 5
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] - 50
        
        # Background 
        pygame.draw.rect(self.screen, (100, 0, 0),
                        (bar_x, bar_y, bar_width, bar_height))
        # Health (color based on health ratio)
        health_color = self.health_color_from_ratio(health_ratio, high=0.5, mid=0.25)
        pygame.draw.rect(self.screen, health_color,
                        (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
    
    def draw_bullet(self, bullet):
        pos = bullet.pos.astype(int)
        
        if self.bullet_sprite is not None:
            # Rotate bullet sprite to face direction
            rotated = pygame.transform.rotate(
                self.bullet_sprite, 
                -np.degrees(bullet.angle) - 90
            )
            rect = rotated.get_rect(center=pos)
            self.screen.blit(rotated, rect)
        else:
            # Fallback
            pygame.draw.circle(self.screen, config.COLOR_BULLET, pos, 5)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 3)
    
    def draw_ui(self, player_health, phase, enemy_count, spawner_count):
        # Semi-transparent background
        ui_bg = pygame.Surface((200, 120))
        ui_bg.set_alpha(128)
        ui_bg.fill((0, 0, 0))
        self.screen.blit(ui_bg, (5, 5))
        
        # Health with color coding
        health_ratio = player_health / max(1.0, float(config.PLAYER_MAX_HEALTH))
        health_color = self.health_color_from_ratio(health_ratio, high=0.6, mid=0.3)
        
        health_text = self.font.render(f"Health: {int(player_health)}", True, health_color)
        phase_text = self.font.render(f"Phase: {phase}", True, config.COLOR_UI)
        enemies_text = self.font.render(f"Enemies: {enemy_count}", True, config.COLOR_UI)
        spawners_text = self.font.render(f"Spawners: {spawner_count}", True, config.COLOR_UI)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(phase_text, (10, 35))
        self.screen.blit(enemies_text, (10, 60))
        self.screen.blit(spawners_text, (10, 85))

    def draw_menu(self, mouse_pos):
        """Draw right-side menu buttons if present."""
        if not self.buttons:
            return

        # Update hover state
        for b in self.buttons.values():
            b.update_hover(mouse_pos)

        # Header
        header = self.font.render("Controls", True, config.COLOR_UI)
        # Draw header box
        x = self.width - 220
        pygame.draw.rect(self.screen, (0, 0, 0), (x - 10, 10, 200, 220))
        self.screen.blit(header, (x + 10, 12))

        # Draw buttons
        for b in self.buttons.values():
            b.draw(self.screen, self.font)

        # Current control label
        try:
            cur = next(k for k, v in self.buttons.items() if v.active)
            label = self.font.render(f"Active: {cur}", True, config.COLOR_UI)
            self.screen.blit(label, (x + 10, 220))
        except StopIteration:
            pass
    
    def update_display(self, fps_scale=1):
        pygame.display.flip()
        # Allow fast-mode scaling when requested
        self.clock.tick(int(config.FPS * fps_scale))
    
    def close(self):
        if self.screen is not None:
            pygame.quit()