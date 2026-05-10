from sim.environment import SmartIrrigationEnv


env = SmartIrrigationEnv()

state = env.reset()

print("Initial State:", state)

action = 1

next_state, reward, done = env.step(action)

print("Action Taken:", action)
print("Next State:", next_state)
print("Reward:", reward)