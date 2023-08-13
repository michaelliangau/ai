# Native imports
import random
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
batch_size = 2
num_epochs = 10

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

    # Run episode
    while not done:
        # Encode state
        state_tensor = torch.tensor(state)
        state_one_hot_encoded = F.one_hot(state_tensor, num_classes=env.state_space.size).float()
        
        # Select action
        action, old_prob = agent.select_action(state_one_hot_encoded)

        # Get environment feedback
        next_state, reward = env.step(action)
        next_state_one_hot_encoded = F.one_hot(torch.tensor(next_state), num_classes=env.state_space.size).float()

        # Store experience in memory
        episode_done = 1 if reward == 1 or reward == -1 else 0
        agent.store_memory(state_one_hot_encoded, action, reward, next_state_one_hot_encoded, episode_done, old_prob)

        # Update state
        state = next_state

        # Terminate episode
        if reward == 1 or reward == -1:
            done = True

    # Shuffle and divide experiences into mini-batches
    experiences = agent.memory
    random.shuffle(experiences)

    # Divide experiences into mini-batches of a specified size
    mini_batches = [experiences[i:i + batch_size] for i in range(0, len(experiences), batch_size)]

    # Take training steps with the agent.
    # In PPO, we train on a single episode's data for multiple epochs. Training is
    # stabilised by clipping surrogate objective.
    for epoch in range(num_epochs):
        for mini_batch in mini_batches:
            states, actions, rewards, next_states, dones, old_probs = zip(*mini_batch)
            agent.step(states, actions, rewards, next_states, dones, old_probs)






