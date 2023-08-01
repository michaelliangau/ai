import environments
import agents

env = environments.GridWorld(grid_size=4, hole_count=4)
agent = agents.DQNAgent(state_size=16, action_size=4)


exit()
# TODO below is unvetted
num_episodes = 1000
epsilon = 0.9
epsilon_min = 0.05
epsilon_decay = 0.995

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward = env.step(action)
        done = (reward != 0)
        agent.store(state, action, reward, next_state, done)
        state = next_state

    agent.train(epsilon)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
