from sim.environment import SmartIrrigationEnv
from sim.q_learning_agent import QLearningAgent

import pandas as pd
import matplotlib.pyplot as plt
import pickle


# ======================================
# EXPERIMENT CONFIGURATION
# ======================================

experiment_name = "q_learning_tuned_v1"


# ======================================
# CREATE ENVIRONMENT AND AGENT
# ======================================

env = SmartIrrigationEnv()

agent = QLearningAgent(
    learning_rate=0.05,
    discount_factor=0.99,
    epsilon=0.10
)


# ======================================
# TRAINING SETTINGS
# ======================================

episodes = 1000
steps_per_episode = 20


# ======================================
# TRACKING REWARDS
# ======================================

episode_rewards = []


# ======================================
# TRAINING LOOP
# ======================================

for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(steps_per_episode):

        action = agent.choose_action(state)

        next_state, reward, done, water_used = env.step(action)

        agent.update_q_table(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:

        average_reward = sum(
            episode_rewards[-100:]
        ) / 100

        print(
            f"Episode {episode + 1} | "
            f"Average Reward: {average_reward:.2f}"
        )


# ======================================
# SAVE Q-TABLE
# ======================================

q_table_path = (
    f"models/{experiment_name}_qtable.pkl"
)

with open(q_table_path, "wb") as file:

    pickle.dump(agent.q_table, file)

print("\nQ-table saved successfully.")


# ======================================
# SAVE RESULTS CSV
# ======================================

results_df = pd.DataFrame({

    "episode": list(range(1, episodes + 1)),
    "reward": episode_rewards
})

results_csv_path = (
    f"experiments/{experiment_name}_results.csv"
)

results_df.to_csv(
    results_csv_path,
    index=False
)

print("Training results saved.")


# ======================================
# PLOT REWARDS
# ======================================

plt.figure(figsize=(12, 6))


# raw rewards

plt.plot(
    episode_rewards,
    alpha=0.4,
    label="Raw Rewards"
)


# moving average

window_size = 50

moving_avg = pd.Series(
    episode_rewards
).rolling(window=window_size).mean()

plt.plot(
    moving_avg,
    linewidth=3,
    label="Moving Average"
)

plt.title(
    f"{experiment_name} Reward Over Episodes"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.legend()


# save plot

plot_path = (
    f"plots/{experiment_name}_plot.png"
)

plt.savefig(plot_path)

plt.show()