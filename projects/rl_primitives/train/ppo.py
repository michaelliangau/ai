# Native imports
# Import packages from parent directory
import sys
sys.path.append("..")

# Third party imports
import IPython
import torch
import torch.nn.functional as F

# Our imports
import environments
import agents

# Hyperparameters
epsilon = 0.2
gamma = 0.99

# Initialization
env = environments.GridWorld(grid_size=4, hole_count=4)
print(f"env", env.grid)
agent = agents.PPOAgent(
    state_size=env.state_space.size,
    action_size=env.action_space.size,
    epsilon=epsilon,
    gamma=gamma,
)


num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # Encode state
        state_tensor = torch.tensor(state)
        state_one_hot_encoded = F.one_hot(state_tensor, num_classes=env.state_space.size).float()
        
        # Select action
        action, old_prob = agent.select_action(state_one_hot_encoded)

        # Get environment feedback
        next_state, reward = env.step(action)
        next_state_one_hot_encoded = F.one_hot(torch.tensor(next_state), num_classes=env.state_space.size).float()

        # Update agent
        step_done = 0 if reward == 1 else 1
        agent.step(state_one_hot_encoded, action, reward, next_state_one_hot_encoded, step_done, old_prob)
        state = next_state

        # Terminate episode
        if reward == 1 or reward == -1:
            done = True






