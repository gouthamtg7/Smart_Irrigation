from sim.environment import SmartIrrigationEnv
from sim.q_learning_agent import QLearningAgent


env = SmartIrrigationEnv()

agent = QLearningAgent()

state = env.reset()

print("Initial State:", state)

action = agent.choose_action(state)

print("Chosen Action:", action)

next_state, reward, done = env.step(action)

print("Reward:", reward)

agent.update_q_table(
    state,
    action,
    reward,
    next_state
)

print("\nUpdated Q-Table:\n")

for key, value in agent.q_table.items():
    print(key, ":", round(value, 2))