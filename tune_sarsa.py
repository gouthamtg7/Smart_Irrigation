from sim.environment import SmartIrrigationEnv
from sim.sarsa_agent import SARSAAgent

import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# HYPERPARAMETER CONFIGURATIONS
# ======================================

configs = [

    {
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.2
    },

    {
        "alpha": 0.05,
        "gamma": 0.99,
        "epsilon": 0.1
    },

    {
        "alpha": 0.2,
        "gamma": 0.90,
        "epsilon": 0.3
    },

    {
        "alpha": 0.15,
        "gamma": 0.95,
        "epsilon": 0.05
    }

]


# ======================================
# TRAINING SETTINGS
# ======================================

episodes = 500
steps_per_episode = 20


# ======================================
# TRACKING RESULTS
# ======================================

tuning_results = []


# ======================================
# RUN EXPERIMENTS
# ======================================

for i, config in enumerate(configs):

    print(f"\nRunning SARSA Experiment {i+1}")

    env = SmartIrrigationEnv()

    agent = SARSAAgent(

        learning_rate=config["alpha"],
        discount_factor=config["gamma"],
        epsilon=config["epsilon"]

    )

    episode_rewards = []

    for episode in range(episodes):

        state = env.reset()

        action = agent.choose_action(state)

        total_reward = 0

        for step in range(steps_per_episode):

            next_state, reward, done, water_used = env.step(action)

            next_action = agent.choose_action(next_state)

            agent.update_q_table(
                state,
                action,
                reward,
                next_state,
                next_action
            )

            state = next_state
            action = next_action

            total_reward += reward

        episode_rewards.append(total_reward)

    avg_reward = sum(
        episode_rewards
    ) / len(episode_rewards)

    tuning_results.append({

        "Experiment": i + 1,

        "Alpha": config["alpha"],
        "Gamma": config["gamma"],
        "Epsilon": config["epsilon"],

        "Average Reward": round(avg_reward, 2)

    })


# ======================================
# CREATE DATAFRAME
# ======================================

results_df = pd.DataFrame(
    tuning_results
)

print("\nSARSA Tuning Results:\n")

print(results_df)


# ======================================
# SAVE RESULTS
# ======================================

results_df.to_csv(
    "experiments/sarsa_tuning.csv",
    index=False
)

print("\nSARSA tuning results saved.")


# ======================================
# PLOT RESULTS
# ======================================

plt.figure(figsize=(10, 6))

experiment_labels = [
    f"Exp {i}"
    for i in results_df["Experiment"]
]

plt.bar(
    experiment_labels,
    results_df["Average Reward"]
)

plt.title(
    "SARSA Hyperparameter Tuning"
)

plt.xlabel("Experiment")
plt.ylabel("Average Reward")

plt.savefig(
    "plots/sarsa_tuning.png"
)

plt.show()