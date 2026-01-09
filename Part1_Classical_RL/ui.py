import pygame
from levels import LEVELS

# UI buttons
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

        # states
        self.active = False     
        self.toggle = False    
        self.enabled = True
        self.hovered = False

    def update_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, screen, font):
        # Background color logic
        if not self.enabled:
            bg = (70, 70, 70)
        elif self.active:
            bg = (70, 140, 220)       # selected
        elif self.toggle:
            bg = (80, 170, 120)       # toggle ON
        elif self.hovered:
            bg = (140, 140, 140)      # hover
        else:
            bg = (110, 110, 110)      # normal

        pygame.draw.rect(screen, bg, self.rect, border_radius=6)
        pygame.draw.rect(screen, (30, 30, 30), self.rect, 2, border_radius=6)

        text_color = (255, 255, 255) if self.enabled else (160, 160, 160)
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def clicked(self, pos):
        return self.enabled and self.rect.collidepoint(pos)
    
# UI layout
def create_ui(tile_w):
    buttons = {}

    # Algorithm selection
    buttons["q"] = Button((tile_w + 40, 30, 160, 35), "Q-Learning")
    buttons["s"] = Button((tile_w + 220, 30, 160, 35), "SARSA")

    # Controls
    buttons["play"] = Button((tile_w + 40, 80, 160, 35), "Play / Pause")
    buttons["fast"] = Button((tile_w + 220, 80, 160, 35), "Fast Mode")

    # Model I/O
    buttons["save"] = Button((tile_w + 40, 130, 160, 35), "Save Model")
    buttons["load"] = Button((tile_w + 220, 130, 160, 35), "Load Model")

    # Intrinsic Reward Toggle (for Level 6)
    buttons["intrinsic"] = Button((tile_w + 220, 325, 180, 35), "Intrinsic Reward")
    buttons["intrinsic"].enabled = False
    
    level_buttons = []
    for i in range(len(LEVELS)):
        x = tile_w + 40 + (i % 2) * 180
        y = 190 + (i // 2) * 45
        level_buttons.append(Button((x, y, 160, 35), f"Level {i}"))

    return buttons, level_buttons