"""
Q-Learning Agent with exponential epsilon decay and improved update logic.
"""

import numpy as np
import random


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy exploration and Bellman equation updates."""

    def __init__(self, num_states, num_actions=4, lr=0.1, gamma=0.9,
                 epsilon=0.9, epsilon_min=0.01, epsilon_decay=0.998,
                 optimistic_init=50.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Optimistic initialization: start Q-values high to encourage exploration
        # of unvisited states. This is critical for sparse-reward maze problems.
        self.q_table = np.full((num_states, num_actions), optimistic_init)
        self.epsilon_history = []

    def choose_action(self, state, valid_actions):
        """
        Epsilon-greedy action selection with random tie-breaking.
        Tie-breaking is critical with optimistic initialization: without it,
        the agent deterministically picks the first action when Q-values are equal,
        creating strong directional bias that prevents maze exploration.
        Returns (action_index, was_exploitation).
        """
        if not valid_actions:
            return 0, True

        if random.random() < self.epsilon:
            q_vals = self.q_table[state][valid_actions]
            max_q = np.max(q_vals)
            # Random tie-breaking among actions with near-max Q-value
            best_candidates = [i for i, idx in enumerate(range(len(valid_actions)))
                               if abs(q_vals[i] - max_q) < 1e-6]
            chosen = random.choice(best_candidates)
            return valid_actions[chosen], True
        else:
            return random.choice(valid_actions), False

    def update(self, state, action, reward, next_state, done):
        """Q-value update using the Bellman equation."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def decay_epsilon(self):
        """Exponential epsilon decay."""
        self.epsilon_history.append(self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def set_eval_mode(self):
        """Switch to evaluation mode: no exploration."""
        self.epsilon = 0.0

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        try:
            self.q_table = np.load(filepath)
            return True
        except FileNotFoundError:
            return False

    def __repr__(self):
        return (f"QLearningAgent(states={self.num_states}, lr={self.lr}, "
                f"gamma={self.gamma}, eps={self.epsilon:.3f}, "
                f"eps_decay={self.epsilon_decay})")
