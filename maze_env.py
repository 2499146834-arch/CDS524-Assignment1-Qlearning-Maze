"""
Maze environment using randomized Prim's algorithm.
Supports regeneration for multi-maze training.
"""

import numpy as np
import random


class Maze:
    """Maze environment: generates perfect mazes with guaranteed path from start to end."""

    def __init__(self, size=10, seed=None):
        self.size = size
        self.start = (0, 0)
        self.end = (size - 1, size - 1)
        self.maze = None
        self._rng = random.Random(seed)
        self.generate()

    def generate(self):
        """Generate a new random maze using randomized Prim's algorithm."""
        self.maze = self._prim_algorithm()
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.end[0]][self.end[1]] = 0

    def _prim_algorithm(self):
        """
        Randomized Prim's algorithm for perfect maze generation.
        Uses the frontier-based approach: every cell is reachable,
        and there is exactly one path between any two cells.
        """
        size = self.size
        maze = np.ones((size, size), dtype=int)

        sx = self._rng.randint(0, size - 1)
        sy = self._rng.randint(0, size - 1)
        maze[sx][sy] = 0

        # Frontier: wall cells adjacent to at least one passage cell
        frontier = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx][ny] == 1:
                frontier.append((nx, ny))

        while frontier:
            idx = self._rng.randrange(len(frontier))
            fx, fy = frontier.pop(idx)

            if maze[fx][fy] == 0:
                continue  # Already carved

            # Find all passage neighbors to connect to
            passage_neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < size and 0 <= ny < size and maze[nx][ny] == 0:
                    passage_neighbors.append((nx, ny))

            if passage_neighbors:
                # Carve this cell, connecting it to the maze
                maze[fx][fy] = 0

                # Add new frontier cells from this newly carved cell
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = fx + dx, fy + dy
                    if 0 <= nx < size and 0 <= ny < size and maze[nx][ny] == 1:
                        if (nx, ny) not in frontier:
                            frontier.append((nx, ny))

        return maze

    def is_valid(self, x, y):
        """Check if (x, y) is within bounds and not a wall."""
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 0

    def get_state_index(self, x, y):
        """Encode (x, y) to a flat state index."""
        return x * self.size + y

    @property
    def num_states(self):
        return self.size * self.size

    def get_valid_actions(self, x, y):
        """Return list of action indices that lead to valid cells from (x, y)."""
        valid = []
        for a, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            if self.is_valid(x + dx, y + dy):
                valid.append(a)
        return valid

    def __repr__(self):
        return f"Maze(size={self.size})"
