import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):

    def __init__(self):

        super(DQN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(3, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 2)

        )

    def forward(self, x):

        return self.network(x)


class DQNAgent:

    def __init__(
        self,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=0.2
    ):

        self.gamma = discount_factor
        self.epsilon = epsilon

        self.model = DQN()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

    def choose_action(self, state):

        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])

        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():

            q_values = self.model(state_tensor)

        return torch.argmax(q_values).item()

    def train_step(
        self,
        state,
        action,
        reward,
        next_state
    ):

        state_tensor = torch.FloatTensor(state)

        next_state_tensor = torch.FloatTensor(
            next_state
        )

        reward_tensor = torch.tensor(reward)

        current_q = self.model(state_tensor)[action]

        with torch.no_grad():

            max_future_q = torch.max(
                self.model(next_state_tensor)
            )

            target_q = reward_tensor + (
                self.gamma * max_future_q
            )

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()