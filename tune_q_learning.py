from sim.environment import SmartIrrigationEnv
from sim.q_learning_agent import QLearningAgent

import pandas as pd


# hyperparameter configurations

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


episodes = 500
steps_per_episode = 20


tuning_results = []


# run experiments

for i, config in enumerate(configs):

    print(f"\nRunning Experiment {i+1}")

    env = SmartIrrigationEnv()

    agent = QLearningAgent(

        learning_rate=config["alpha"],
        discount_factor=config["gamma"],
        epsilon=config["epsilon"]

    )

    episode_rewards = []

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

    avg_reward = sum(episode_rewards) / len(episode_rewards)

    tuning_results.append({

        "Experiment": i + 1,

        "Alpha": config["alpha"],
        "Gamma": config["gamma"],
        "Epsilon": config["epsilon"],

        "Average Reward": round(avg_reward, 2)

    })


# convert to dataframe

results_df = pd.DataFrame(tuning_results)

print("\nTuning Results:\n")

print(results_df)


# save results

results_df.to_csv(
    "experiments/q_learning_tuning.csv",
    index=False
)

print("\nTuning results saved.")