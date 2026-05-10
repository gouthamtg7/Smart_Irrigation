from sim.environment import SmartIrrigationEnv
from sim.dqn_agent import DQNAgent

import pandas as pd
import matplotlib.pyplot as plt

import torch


env = SmartIrrigationEnv()

agent = DQNAgent()


episodes = 1000
steps_per_episode = 20

episode_rewards = []


for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(steps_per_episode):

        action = agent.choose_action(state)

        next_state, reward, done, water_used = env.step(action)

        agent.train_step(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:

        avg_reward = sum(
            episode_rewards[-100:]
        ) / 100

        print(
            f"Episode {episode+1} | "
            f"Average Reward: {avg_reward:.2f}"
        )


# save model

torch.save(
    agent.model.state_dict(),
    "models/dqn_model.pth"
)


# save results

results_df = pd.DataFrame({

    "episode": list(range(1, episodes + 1)),
    "reward": episode_rewards
})

results_df.to_csv(
    "experiments/dqn_results.csv",
    index=False
)


# plot

plt.figure(figsize=(12, 6))

plt.plot(
    episode_rewards,
    alpha=0.4,
    label="Raw Rewards"
)

moving_avg = pd.Series(
    episode_rewards
).rolling(window=50).mean()

plt.plot(
    moving_avg,
    linewidth=3,
    label="Moving Average"
)

plt.title("DQN Reward Over Episodes")

plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend()

plt.savefig(
    "plots/dqn_reward_plot.png"
)

plt.show()