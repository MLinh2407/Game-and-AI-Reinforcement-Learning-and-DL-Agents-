import numpy as np
import config


class Player:
    def __init__(self, x, y, control_scheme="rotation"):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.angle = -np.pi / 2
        self.health = config.PLAYER_MAX_HEALTH
        self.max_health = config.PLAYER_MAX_HEALTH
        self.control_scheme = control_scheme

        # Shooting
        self.last_shot_time = 0
        self.shoot_cooldown = config.PLAYER_SHOOT_COOLDOWN

    def reset(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.angle = -np.pi / 2
        self.health = config.PLAYER_MAX_HEALTH
        self.last_shot_time = 0

    # Apply thrust forward (rotation control)
    def move_thrust(self):
        thrust = config.PLAYER_THRUST_SPEED
        self.vel[0] += thrust * np.cos(self.angle)
        self.vel[1] += thrust * np.sin(self.angle)

    # Move in specific direction (directional control)
    def move_direction(self, direction):
        speed = config.PLAYER_DIRECT_SPEED
        if direction == "up":
            self.vel[1] = -speed
        elif direction == "down":
            self.vel[1] = speed
        elif direction == "left":
            self.vel[0] = -speed
        elif direction == "right":
            self.vel[0] = speed

    # Rotate ship left or right
    def rotate(self, direction):
        if direction == "left":
            self.angle -= config.PLAYER_ROTATION_SPEED
        elif direction == "right":
            self.angle += config.PLAYER_ROTATION_SPEED

    # Update player position and velocity
    def update(self, bounds):
        # Apply velocity
        self.pos += self.vel

        # Apply friction based on control scheme
        if self.control_scheme == config.CONTROL_ROTATION:
            self.vel *= config.PLAYER_FRICTION_ROTATION
        else:
            self.vel *= config.PLAYER_FRICTION_DIRECT
            self.angle = -np.pi / 2

        # Keep player in bounds
        margin = 20
        self.pos[0] = np.clip(self.pos[0], margin, bounds[0] - margin)
        self.pos[1] = np.clip(self.pos[1], margin, bounds[1] - margin)

    # Check if player can shoot based on cooldown
    def can_shoot(self, current_time):
        return current_time - self.last_shot_time >= self.shoot_cooldown

    # Get bullet velocity based on control scheme
    def shoot_velocity(self, target_pos=None):
        if self.control_scheme == config.CONTROL_ROTATION:
            # Shoot in direction of angle
            vel = np.array(
                [
                    config.BULLET_SPEED * np.cos(self.angle),
                    config.BULLET_SPEED * np.sin(self.angle),
                ]
            )
            angle = self.angle
        else:
            # Directional control: bullets always fire straight up
            # in world coordinates. The ship's angle remains fixed;
            # only its position changes.
            vel = np.array([0.0, -config.BULLET_SPEED])
            angle = -np.pi / 2

        return vel, angle

    # Take damage and return True if still alive
    def take_damage(self, damage):
        self.health -= damage
        return self.health > 0

    # Check if player is alive
    def is_alive(self):
        return self.health > 0

    # Get position for thrust particle (behind ship)
    def get_thrust_particle_pos(self):
        offset_x = -15 * np.cos(self.angle)
        offset_y = -15 * np.sin(self.angle)
        return self.pos + np.array([offset_x, offset_y])