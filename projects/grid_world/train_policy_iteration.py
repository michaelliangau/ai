import environments
import agents
import numpy as np

np.random.seed(0)  # For reproducibility


# Map action to direction
action_to_direction = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# Initialize agent and environments
env = environments.GridWorld(grid_size=4, hole_count=1)
agent = agents.PolicyIterationAgent(
    num_states=env.state_space.shape[0],
    goal_state_idx=env.state_space.shape[0] - 1,  # final state is goal state
    num_actions=env.action_space.shape[0],
)

# Train agent
agent.policy_iteration(env)


# Traverse the grid
current_state = 0

# Keep track of the states and actions taken for visualization
states = [current_state]
actions = []

# Traverse the grid based on the policy until we reach the goal
while current_state != agent.goal_state_idx:
    # Get the action for the current state from the policy
    action = agent.get_action(current_state)

    # Store action
    actions.append(action_to_direction[action])

    # Get the next state and done flag from the environment
    prob, next_state, reward, done = env.transitions(current_state, action)[0]

    # Store the next state
    states.append(next_state)

    # Update the current state
    current_state = next_state

# Print the states and actions
for state, action in zip(states, actions):
    print(f"State: {state}, Action: {action}")

print(f"Final State: {states[-1]}")
