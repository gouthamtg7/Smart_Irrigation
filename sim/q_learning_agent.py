import random
import numpy as np


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2
    ):

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        self.q_table = {}

    def get_q_value(self, state, action):

        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0

        return self.q_table[(state, action)]

    def choose_action(self, state):

        # exploration

        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])

        # exploitation

        q_no_irrigation = self.get_q_value(state, 0)
        q_irrigation = self.get_q_value(state, 1)

        if q_no_irrigation > q_irrigation:
            return 0

        return 1

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state
    ):

        current_q = self.get_q_value(state, action)

        max_future_q = max(
            self.get_q_value(next_state, 0),
            self.get_q_value(next_state, 1)
        )

        new_q = current_q + self.alpha * (
            reward +
            self.gamma * max_future_q -
            current_q
        )

        self.q_table[(state, action)] = new_q