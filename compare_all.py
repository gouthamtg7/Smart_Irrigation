import pandas as pd
import matplotlib.pyplot as plt


# load results

q_learning_results = pd.read_csv(
    "experiments/ql_results.csv"
)

sarsa_results = pd.read_csv(
    "experiments/sarsa_results.csv"
)

dqn_results = pd.read_csv(
    "experiments/dqn_results.csv"
)


# moving averages

window_size = 50

q_ma = q_learning_results[
    "reward"
].rolling(window=window_size).mean()

sarsa_ma = sarsa_results[
    "reward"
].rolling(window=window_size).mean()

dqn_ma = dqn_results[
    "reward"
].rolling(window=window_size).mean()


# plot comparison

plt.figure(figsize=(14, 7))

plt.plot(
    q_ma,
    label="Q-Learning",
    linewidth=3
)

plt.plot(
    sarsa_ma,
    label="SARSA",
    linewidth=3
)

plt.plot(
    dqn_ma,
    label="DQN",
    linewidth=3
)

plt.title(
    "RL Algorithm Comparison"
)

plt.xlabel("Episode")
plt.ylabel("Moving Average Reward")

plt.legend()

plt.savefig(
    "plots/final_algorithm_comparison.png"
)

plt.show()


# calculate averages

q_avg = q_learning_results[
    "reward"
].mean()

sarsa_avg = sarsa_results[
    "reward"
].mean()

dqn_avg = dqn_results[
    "reward"
].mean()


# comparison table

comparison_df = pd.DataFrame({

    "Algorithm": [
        "Q-Learning",
        "SARSA",
        "DQN"
    ],

    "Average Reward": [
        round(q_avg, 2),
        round(sarsa_avg, 2),
        round(dqn_avg, 2)
    ]
})


print("\nFinal Algorithm Comparison:\n")

print(comparison_df)