import environments
import agents
import numpy as np

np.random.seed(0)  # For reproducibility

# Map action to direction
action_to_direction = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# Initialize agent and environments
env = environments.GridWorld(grid_size=4, hole_count=4)
print(env.grid)
agent = agents.QLearningAgent(env.state_space.shape[0], env.action_space.shape[0])

# Training parameters
num_episodes = 10000
max_steps_per_episode = 100

for episode in range(num_episodes):
    # Reset the state, env doesn't change
    state = env.reset()

    for step in range(max_steps_per_episode):
        action = agent.get_epsilon_greedy_action(state)
        next_state, reward = env.step(action)
        agent.update_Q(state, action, reward, next_state)

        state = next_state

        if reward == -1 or reward == 1:  # agent fell in a hole or reached the goal
            break

    # Print out progress
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed")

print("Training finished.")
print("Q table:")
print(agent.Q)
