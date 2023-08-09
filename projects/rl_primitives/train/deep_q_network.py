# Native imports
# Import packages from parent directory
import sys

sys.path.append("..")

# Third party imports
import IPython
from tqdm import tqdm

# Our imports
import environments
import agents

# Hyperparameters
num_episodes = 100000
epsilon = 0.9
epsilon_min = 0.05
epsilon_decay = 0.999995
target_q_net_update_freq = 10
print_freq = 1000
lr = 3e-5

# Initialization
env = environments.GridWorld(grid_size=4, hole_count=4)
print(f"env", env.grid)
agent = agents.DQNAgent(
    state_size=16,
    action_size=4,
    batch_size=4,
    target_q_net_update_freq=target_q_net_update_freq,
    lr=lr,
)


# Training loop
losses = []
for num_step in tqdm(range(num_episodes)):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward = env.step(action)
        done = reward != 0
        agent.store(state, action, reward, next_state, done)
        state = next_state

    loss = agent.train(num_step=num_step)
    losses.append(loss)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print logs
    if num_step % print_freq == 0:
        print(f"num_step: {num_step}")
        print(f"epsilon: {epsilon}")
        print(f"Mean last print_freq loss: {sum(losses[-print_freq:]) / print_freq}")
