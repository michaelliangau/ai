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


# Initialization
env = environments.GridWorld(grid_size=4, hole_count=4)
print(f"env", env.grid)
agent = agents.PPOAgent(
    state_size=env.state_space.size,
    action_size=env.action_space.size,
)


num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state)
        state_one_hot_encoded = F.one_hot(state_tensor, num_classes=env.state_space.size).float()
        action = agent.select_action(state_one_hot_encoded)
        next_state, reward = env.step(action)
        # TODO WIP up to here
        old_prob = agent.policy_network(torch.FloatTensor(state)).detach()[action]
        agent.step(state, action, reward, next_state, 0 if reward == 1 else 1, old_prob)
        state = next_state
        if reward == 1 or reward == -1:
            done = True






