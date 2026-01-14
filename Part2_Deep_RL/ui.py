import pygame
import config

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


def create_ui(width):
    """Create a simple right-side UI with control scheme buttons and a fast-mode toggle."""
    buttons = {}

    x = width - 195
    # Control scheme buttons
    buttons["rotation"] = Button((x, 30, 180, 36), "Rotation")
    buttons["directional"] = Button((x, 80, 180, 36), "Directional")

    # Fast-mode toggle (speeds rendering/step rate)
    buttons["fast"] = Button((x, 130, 180, 36), "Fast Mode")

    # Human control toggle
    buttons["human"] = Button((x, 180, 180, 36), "Human Control")
    buttons["human"].toggle = False

    # Start with rotation active by default
    if config.CONTROL_ROTATION:
        buttons["rotation"].active = True

    return buttons