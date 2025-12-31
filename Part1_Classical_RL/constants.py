# =========================
# Grid & Tile Configuration
# =========================

TILE_SIZE = 80

# Tile IDs
FLOOR   = 0
ROCK    = 1
APPLE   = 2
FIRE    = 3
KEY     = 4
CHEST   = 5
MONSTER = 6

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = {
    UP:    (0, -1),
    DOWN:  (0,  1),
    LEFT:  (-1, 0),
    RIGHT: (1,  0)
}
