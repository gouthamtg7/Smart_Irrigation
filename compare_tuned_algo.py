import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# LOAD TUNED RESULTS
# ======================================

q_learning_results = pd.read_csv(
    "experiments/q_learning_tuned_v1_results.csv"
)

sarsa_results = pd.read_csv(
    "experiments/sarsa_tuned_v1_results.csv"
)


# ======================================
# MOVING AVERAGES
# ======================================

window_size = 50

q_ma = q_learning_results[
    "reward"
].rolling(window=window_size).mean()

sarsa_ma = sarsa_results[
    "reward"
].rolling(window=window_size).mean()


# ======================================
# PLOT COMPARISON
# ======================================

plt.figure(figsize=(14, 7))

plt.plot(
    q_ma,
    label="Tuned Q-Learning",
    linewidth=3
)

plt.plot(
    sarsa_ma,
    label="Tuned SARSA",
    linewidth=3
)

plt.title(
    "Tuned RL Algorithm Comparison"
)

plt.xlabel("Episode")
plt.ylabel("Moving Average Reward")

plt.legend()

plt.savefig(
    "plots/tuned_algorithm_comparison.png"
)

plt.show()


# ======================================
# CALCULATE FINAL AVERAGES
# ======================================

q_avg = q_learning_results[
    "reward"
].mean()

sarsa_avg = sarsa_results[
    "reward"
].mean()


# ======================================
# COMPARISON TABLE
# ======================================

comparison_df = pd.DataFrame({

    "Algorithm": [
        "Tuned Q-Learning",
        "Tuned SARSA"
    ],

    "Average Reward": [
        round(q_avg, 2),
        round(sarsa_avg, 2)
    ]
})


print("\nFinal Tuned Algorithm Comparison:\n")

print(comparison_df)