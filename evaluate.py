from sim.environment import SmartIrrigationEnv
from sim.q_learning_agent import QLearningAgent

import pickle
import pandas as pd


# create environment

env = SmartIrrigationEnv()


# load trained Q-table

agent = QLearningAgent()

with open("models/q_table.pkl", "rb") as file:
    agent.q_table = pickle.load(file)


# evaluation settings

episodes = 100
steps_per_episode = 20


# =========================
# FIXED IRRIGATION BASELINE
# =========================

fixed_total_reward = 0
fixed_total_water = 0


for episode in range(episodes):

    state = env.reset()

    for step in range(steps_per_episode):

        action = 1

        next_state, reward, done, water_used = env.step(action)

        fixed_total_reward += reward
        fixed_total_water += water_used

        state = next_state


# =========================
# RL POLICY EVALUATION
# =========================

rl_total_reward = 0
rl_total_water = 0


for episode in range(episodes):

    state = env.reset()

    for step in range(steps_per_episode):

        action = agent.choose_action(state)

        next_state, reward, done, water_used = env.step(action)

        rl_total_reward += reward
        rl_total_water += water_used

        state = next_state


# calculate averages

fixed_avg_reward = fixed_total_reward / episodes
rl_avg_reward = rl_total_reward / episodes

fixed_avg_water = fixed_total_water / episodes
rl_avg_water = rl_total_water / episodes


# comparison table

results = pd.DataFrame({

    "Method": [
        "Fixed Irrigation",
        "Q-Learning RL"
    ],

    "Average Reward": [
        fixed_avg_reward,
        rl_avg_reward
    ],

    "Average Water Usage": [
        fixed_avg_water,
        rl_avg_water
    ]
})


print("\nEvaluation Results:\n")

print(results)