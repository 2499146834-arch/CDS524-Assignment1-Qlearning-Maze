"""
Headless trainer with full metric collection.
Supports fixed-maze and random-maze training modes.
"""

import numpy as np
from collections import deque

from maze_env import Maze
from ql_agent import QLearningAgent


class Trainer:
    """Trains a Q-learning agent on a maze and collects per-episode metrics."""

    def __init__(self, maze_size=10, maze_mode='fixed',
                 lr=0.1, gamma=0.9, epsilon=0.9,
                 epsilon_decay=0.998, epsilon_min=0.01,
                 step_penalty=-1, revisit_penalty=-5,
                 max_steps=200, consecutive_success=50,
                 optimal_step_threshold=None, optimistic_init=50.0,
                 distance_reward=0.0, seed=None):
        self.maze_size = maze_size
        self.maze_mode = maze_mode
        self.max_steps = max_steps
        self.consecutive_success = consecutive_success
        self.optimal_step_threshold = optimal_step_threshold or (maze_size * 2)
        self.seed = seed

        # Reward parameters
        self.step_penalty = step_penalty
        self.revisit_penalty = revisit_penalty
        self.distance_reward = distance_reward

        # Create maze and agent
        self.maze = Maze(maze_size, seed=seed)
        self.agent = QLearningAgent(
            num_states=maze_size * maze_size,
            lr=lr, gamma=gamma,
            epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            optimistic_init=optimistic_init
        )

        # Episode state
        self.agent_x = self.agent_y = 0
        self.visited = set()
        self.step = 0
        self.cumulative_reward = 0
        self.episode_done = False

        # Metrics
        self.metrics = []
        self._consecutive_count = 0

    def _calculate_reward(self, nx, ny):
        """Reward function with optional distance-based shaping (no double-counting bug)."""
        if (nx, ny) == self.maze.end:
            return 100.0
        if not self.maze.is_valid(nx, ny):
            return -10.0
        if (nx, ny) in self.visited:
            return float(self.revisit_penalty)

        reward = float(self.step_penalty)

        # Distance-based reward shaping: bonus for moving closer to goal
        if self.distance_reward != 0.0:
            ex, ey = self.maze.end
            cur_dist = abs(self.agent_x - ex) + abs(self.agent_y - ey)
            new_dist = abs(nx - ex) + abs(ny - ey)
            reward += (cur_dist - new_dist) * self.distance_reward

        return reward

    def _reset_episode(self):
        """Reset agent to start for a new episode."""
        if self.maze_mode == 'random':
            self.maze.generate()

        self.agent_x, self.agent_y = self.maze.start
        self.visited = {(self.agent_x, self.agent_y)}
        self.step = 0
        self.cumulative_reward = 0.0
        self.episode_done = False

    def _step(self):
        """Execute one step within an episode. Returns True if episode ended."""
        state = self.maze.get_state_index(self.agent_x, self.agent_y)
        valid_actions = self.maze.get_valid_actions(self.agent_x, self.agent_y)

        action, _ = self.agent.choose_action(state, valid_actions)

        # Compute next coordinates
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        nx, ny = self.agent_x + dx, self.agent_y + dy

        # Calculate reward exactly once
        reward = self._calculate_reward(nx, ny)
        self.cumulative_reward += reward
        self.step += 1

        # Determine if episode ends
        reached_goal = (nx, ny) == self.maze.end
        hit_step_limit = self.step >= self.max_steps

        # Update position if valid
        next_state = state
        if self.maze.is_valid(nx, ny):
            self.agent_x, self.agent_y = nx, ny
            self.visited.add((nx, ny))
            next_state = self.maze.get_state_index(nx, ny)

        # Q-table update
        self.agent.update(state, action, reward, next_state, reached_goal or hit_step_limit)

        if reached_goal or hit_step_limit:
            self.episode_done = True
            return True
        return False

    def _run_episode(self):
        """Run a single episode. Returns metrics dict."""
        self._reset_episode()

        while not self.episode_done:
            self._step()

        success = (self.agent_x, self.agent_y) == self.maze.end
        optimal = success and self.step <= self.optimal_step_threshold

        # Update consecutive success counter (any success counts, not just optimal)
        if success:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        # Decay epsilon after each episode
        self.agent.decay_epsilon()

        return {
            'episode': len(self.metrics) + 1,
            'steps': self.step,
            'reward': self.cumulative_reward,
            'success': success,
            'optimal': optimal,
            'epsilon': self.agent.epsilon_history[-1],
            'consecutive_success': self._consecutive_count,
        }

    def train(self, max_episodes=5000, verbose=True, verbose_interval=200):
        """
        Train the agent for up to max_episodes.
        Returns list of per-episode metrics dicts.
        Early-stops when consecutive_success threshold is met (fixed maze only).
        """
        self.metrics = []
        self._consecutive_count = 0

        for _ in range(max_episodes):
            result = self._run_episode()
            self.metrics.append(result)

            if verbose and result['episode'] % verbose_interval == 0:
                print(f"Ep {result['episode']:5d} | "
                      f"Steps: {result['steps']:3d} | "
                      f"Reward: {result['reward']:7.1f} | "
                      f"Success: {result['success']} | "
                      f"Optimal: {result['optimal']} | "
                      f"Epsilon: {result['epsilon']:.3f} | "
                      f"Consec: {result['consecutive_success']}")

            # Early stop: N consecutive successes AND epsilon has decayed enough
            if (self.maze_mode == 'fixed'
                    and self._consecutive_count >= self.consecutive_success
                    and self.agent.epsilon <= 0.1):
                if verbose:
                    print(f"\nTraining converged at episode {result['episode']} — "
                          f"{self.consecutive_success} consecutive optimal solves achieved.")
                break

        if verbose and self._consecutive_count < self.consecutive_success and self.maze_mode == 'fixed':
            print(f"\nTraining stopped at max episodes {max_episodes} without full convergence.")

        return self.metrics

    def evaluate(self, num_episodes=100, verbose=False):
        """
        Evaluate trained agent (epsilon=0, no exploration).
        Returns dict with average steps, success rate, optimal rate.
        """
        old_epsilon = self.agent.epsilon
        old_decay = self.agent.epsilon_decay
        self.agent.set_eval_mode()

        total_steps = 0
        successes = 0
        optimals = 0
        eval_mazes = []

        for i in range(num_episodes):
            self._reset_episode()
            while not self.episode_done:
                self._step()

            if (self.agent_x, self.agent_y) == self.maze.end:
                successes += 1
                if self.step <= self.optimal_step_threshold:
                    optimals += 1
            total_steps += self.step
            eval_mazes.append(self.maze.maze.copy() if self.maze_mode == 'random' else None)

        # Restore agent state
        self.agent.epsilon = old_epsilon
        self.agent.epsilon_decay = old_decay

        return {
            'avg_steps': total_steps / num_episodes,
            'success_rate': successes / num_episodes,
            'optimal_rate': optimals / num_episodes,
        }

    def get_metrics_dataframe(self):
        """Return metrics as a pandas DataFrame (if available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.metrics)
        except ImportError:
            return self.metrics


def run_training(maze_size=10, maze_mode='fixed', lr=0.1, gamma=0.9,
                 epsilon=0.9, epsilon_decay=0.998, epsilon_min=0.01,
                 step_penalty=-1, max_episodes=5000, optimistic_init=50.0,
                 seed=42, verbose=True):
    """Convenience function: train and return (trainer, metrics_df)."""
    trainer = Trainer(
        maze_size=maze_size, maze_mode=maze_mode,
        lr=lr, gamma=gamma, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        step_penalty=step_penalty, optimistic_init=optimistic_init,
        seed=seed
    )
    trainer.train(max_episodes=max_episodes, verbose=verbose)
    df = trainer.get_metrics_dataframe()
    return trainer, df
