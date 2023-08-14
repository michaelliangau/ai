import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import random
from collections import deque
import numpy as np
import IPython
from collections import defaultdict
from tqdm import tqdm


class QLearningAgent:
    """Simple agent that implements the Q-learning algorithm.

    It basically maintains a Q table and then updates it according to TD error for each
    action. One action at a time.
    """

    def __init__(
        self, num_states=16, num_actions=4, alpha=0.9, gamma=0.95, epsilon=0.5
    ):
        """Init the agent.

        Args:
            num_states (int): number of states. Defaults to 16 (4 x 4 grid).
            num_actions (int): number of actions. Defaults to 4 (left, down, right, up).
            alpha (float, optional): learning rate. Defaults to 0.5.
            gamma (float, optional): discount factor. Defaults to 0.95.
            epsilon (float, optional): exploration rate. Defaults to 0.1.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q table: Expected return (not reward) for each state-action pair
        self.Q = np.zeros((num_states, num_actions))

    def get_epsilon_greedy_action(self, state):
        """Pick a random action with probability epsilon, otherwise pick the best action

        Args:
            state (int): current state
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # explore
        else:
            # exploit
            # find the indices of the maximum values
            max_indices = np.where(self.Q[state] == np.amax(self.Q[state]))[0]
            # choose randomly from those indices
            return np.random.choice(max_indices)

    def update_Q(self, state, action, reward, next_state):
        """Update the Q table.

        The Q-table is the expected return for each state-action pair.

        This is the core mechanics of Q-learning and implements the Bellman optimality
        equation for Q-values. This is a subset of the general Bellman equation.

        Reference principle of optimality on why Q learning works. Optimal policy of
        subsequences is also optimal for the original sequence.

        Args:
            state (int): current state
            action (int): action taken
            reward (int): reward received
            next_state (int): next state
        """
        # Calculate temporal difference target (TD target). Expected future returns based
        # off a combination of received reward and the expected future returns from
        # best action in the next state.
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]

        # Calculate temporal difference error (TD error). Difference between the
        # implied expected return from received reward and the current Q value prior to
        # update.
        td_error = td_target - self.Q[state][action]

        # Update the Q table value, multiplying the learning rate (alpha) by the error.
        self.Q[state][action] += self.alpha * td_error


class ValueIterationAgent:
    """Value iteration is a model based RL method.

    It basically loops through the states and actions and computes the maximum possible
    return for each state and then updates the value for that state. It stops when
    the magnitude of the value updated drops below a certain theta value.
    """

    def __init__(self, num_states, goal_state_idx, num_actions, theta=1e-8, gamma=0.95):
        """Initialize the ValueIterationAgent.

        Args:
            num_states (int): Number of states in the environment.
            goal_state_idx (int): Index of the goal state.
            num_actions (int): Number of possible actions in the environment.
            theta (float, optional): Threshold for determining the convergence of the value function. Defaults to 1e-8.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.95.
        """
        self.num_states = num_states
        self.goal_state_idx = goal_state_idx
        self.num_actions = num_actions
        self.theta = (
            theta  # small number threshold to determine convergence of value function
        )
        self.gamma = gamma  # discount factor
        self.values = np.zeros(num_states)  # initialize value function
        self.policy = np.zeros(num_states)  # initialize policy

    def value_iteration(self, env):
        """Perform the value iteration algorithm.

        This is the core value iteration mechanic. It gets the maximum value of each
        state by maxing over the possible actions at each state.

        Value iteration assumes you have access to the underlying MDP and can get reward
        values during iteration.

        Values in the value function is expected total reward an agent can expect to
        receive from that state onward, excluding immediate rewards from transitioning
        to the state. Therefore goal states have a value of 0.

        Args:
            env (GridWorld): The environment in which the agent interacts.
        """

        # Infinite loop until convergence.
        while True:
            delta = 0
            for state in range(self.num_states):
                # Skip goal state
                if state == self.goal_state_idx:
                    continue
                # Get current value of state
                state_value = self.values[state]

                # Compute returns for all possible actions at this state
                action_values = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    # Loop over possible transitions (GridWorld has just 1 transition as it is deterministic, doesn't have to be the case)
                    for prob, next_state, reward, _ in env.transitions(state, action):
                        # Compute action values for each action in this state
                        action_values[action] += prob * (
                            reward + self.gamma * self.values[next_state]
                        )

                # Set new value of this state to the maximum action value
                self.values[state] = np.max(action_values)

                # Update delta (how large was the update)
                delta = max(delta, np.abs(state_value - self.values[state]))

            # Break loop once max delta across all states is smaller than theta
            if delta < self.theta:
                break

        # Output a deterministic policy
        for s in range(self.num_states):
            if s == self.goal_state_idx:
                continue
            action_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                for prob, next_state, reward, _ in env.transitions(s, a):
                    action_values[a] += prob * (
                        reward + self.gamma * self.values[next_state]
                    )
            self.policy[s] = np.argmax(action_values)

    def get_action(self, state):
        """Get the action to take in a given state.

        Args:
            state (int): The current state.

        Returns:
            int: The action to take.
        """
        return self.policy[state]


class PolicyIterationAgent:
    """Policy iteration is a model based RL method.

    It works in 2 stages:

    Policy evaluation: Updates the value function based on a current policy until
    the value function estimate stabilizes.

    Policy improvement: Updates the policy based on the current value function.
    This generates a new policy that is guaranteed to be better than or as good as
    the old policy
    """

    def __init__(self, num_states, goal_state_idx, num_actions, theta=1e-8, gamma=0.95):
        """Initialize the PolicyIterationAgent.

        Args:
            num_states (int): Number of states in the environment.
            goal_state_idx (int): Index of the goal state.
            num_actions (int): Number of possible actions in the environment.
            theta (float, optional): Threshold for determining the convergence of the
            value function. Defaults to 1e-8.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.95.
        """
        self.num_states = num_states
        self.goal_state_idx = goal_state_idx
        self.num_actions = num_actions
        self.theta = theta
        self.gamma = gamma
        self.policy = np.zeros(num_states)
        self.values = np.zeros(num_states)

    def policy_evaluation(self, env):
        """Evaluate the current policy.

        This is the core policy evaluation mechanic. It's sort of like value iteration
        except it only updates the value function for the current policy (actions chosen
        by current policy).

        Args:
            env (GridWorld): The environment in which the agent interacts.
        """
        while True:
            delta = 0
            for state in range(self.num_states):
                if state == self.goal_state_idx:
                    continue
                v = self.values[state]
                action = self.policy[state]
                # Value of the state is the expected return from the current action
                # under the current policy
                self.values[state] = sum(
                    [
                        prob * (reward + self.gamma * self.values[next_state])
                        for prob, next_state, reward, _ in env.transitions(
                            state, action
                        )
                    ]
                )
                delta = max(delta, np.abs(v - self.values[state]))
            if delta < self.theta:
                break

    def policy_improvement(self, env):
        """Improve the current policy.

        Evaluates all possible action values for each state similar to value iteration
        and returns whether or not the policy is stable.

        Args:
            env (GridWorld): The environment in which the agent interacts.

        Returns:
            bool: Whether or not the policy is stable.
        """
        policy_stable = True
        for state in range(self.num_states):
            if state == self.goal_state_idx:
                continue
            old_action = self.policy[state]
            action_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                action_values[action] = sum(
                    [
                        prob * (reward + self.gamma * self.values[next_state])
                        for prob, next_state, reward, _ in env.transitions(
                            state, action
                        )
                    ]
                )
            self.policy[state] = np.argmax(action_values)
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def policy_iteration(self, env):
        """Perform the policy iteration algorithm.

        Args:
            env (GridWorld): The environment in which the agent interacts.
        """
        while True:
            self.policy_evaluation(env)
            policy_stable = self.policy_improvement(env)
            if policy_stable:
                break

    def get_action(self, state):
        """Get the action to take in a given state.

        Args:
            state (int): The current state.

        Returns:
            int: The action to take.
        """
        return self.policy[state]


class MonteCarloAgent:
    """Monte Carlo methods are model free RL methods.

    It basically runs episodes and then updates the Q table based on observed returns
    for all the states that were experienced in the episode. It stores the discounted
    returns for each state and averages over them in updating.
    """

    def __init__(self, num_states, num_actions, gamma=0.95, epsilon=0.1):
        """Initialize the MonteCarloAgent.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions in the environment.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.95.
            epsilon (float, optional): Exploration rate. Defaults to 0.1.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = defaultdict(
            lambda: defaultdict(list)
        )  # dict to save returns for each state-action pair
        self.Q = defaultdict(lambda: np.zeros(num_actions))  # action value function
        self.policy = np.zeros(num_states, dtype=int)  # policy

    def generate_episode(self, env):
        """Generate an episode by following the epsilon-greedy policy.

        Args:
            env (GridWorld): The environment in which the agent interacts.

        Returns:
            episode (list): A list of (state, action, reward) tuples
        """
        state = env.reset()
        episode = []
        while True:
            probs = (
                np.ones(self.num_actions, dtype=float) * self.epsilon / self.num_actions
            )
            probs[self.policy[state]] += 1.0 - self.epsilon
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward = env.step(action)
            episode.append((state, action, reward))
            if (
                next_state == env.state_space[-1] or reward == -1 or reward == 1
            ):  # If the last state is reached
                break
            state = next_state
        return episode

    def first_visit_mc_prediction(self, num_episodes, env):
        """First visit Monte Carlo prediction.

        First visit means we only consider the first visit to each state in each
        episode when calculating the average return.

        The core Q table update algorithm is in the for loop iterating through states.

        Args:
            num_episodes (int): Number of episodes to run.
            env (GridWorld): The environment in which the agent interacts.
        """
        for _ in tqdm(range(num_episodes)):
            # Generate an episode
            episode = self.generate_episode(env)

            # Prepare lists
            states, actions, rewards = zip(*episode)
            discounts = np.array([self.gamma**i for i in range(len(rewards) + 1)])

            # Iterate over each state in the episode
            for i, state in enumerate(states):
                # Calculate the return (discounted sum of rewards) from this state onwards
                action = actions[i]
                future_rewards = rewards[i:]
                discounted_future_rewards = discounts[: -(1 + i)] * future_rewards
                return_from_this_state = sum(discounted_future_rewards)

                # Append the calculated return to the list of returns for this state-action pair
                self.returns[state][action].append(return_from_this_state)

                # Average the returns to calculate the action-value for this state-action pair
                average_return = np.mean(self.returns[state][action])
                self.Q[state][action] = average_return

                # Update the policy for this state to be the action with the highest action-value
                best_action_for_this_state = np.argmax(self.Q[state])
                self.policy[state] = best_action_for_this_state


class QNetwork(nn.Module):
    """QNetwork class, state action representation."""

    def __init__(self, state_size, action_size):
        """Init the QNetwork.

        Q network: Input a state, output the Q value for each action.

        Multi-layer fully connected network with ReLU activation and layer normalization.

        Args:
            state_size (int): number of states
            action_size (int): number of actions
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        self.fc5 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.relu(self.ln4(self.fc4(x)))
        x = self.fc5(x)
        return x


class DQNAgent:
    """Deep Q Network agent, uses QNetwork class

    An agent generates a running memory buffer of experiences by traversing the
    environment and then it samples a batch of experiences from the memory buffer
    and uses them to train the Q network.

    It pushes each experience through the q networks to compute
    expected Q values. Then it pushes next state through a target network and uses Bellman
    equation to compute a target Q value. TD error style.
    """

    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lr=0.001,
        batch_size=64,
        memory_size=10000,
        target_q_net_update_freq=10,
    ):
        """Init the QNetwork.

        Args:
            state_size (int): number of states
            action_size (int): number of actions
            gamma (float, optional): discount factor. Defaults to 0.99.
            lr (float, optional): learning rate. Defaults to 0.001.
            batch_size (int, optional): batch size. Defaults to 64.
            memory_size (int, optional): size of replay buffer. Defaults to 10000.
            target_q_net_update_freq (int, optional): how often to update the target
                network in steps. Defaults to 10.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        # deque discards memories once it reaches the max length, ensuring we have
        # the most recent memories
        self.memory = deque(maxlen=memory_size)
        self.target_q_net_update_freq = target_q_net_update_freq

        # Init the Q network and target network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)

        # Copy the weights from the Q network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        print(
            "Number of parameters in Q Net:",
            sum(p.numel() for p in self.q_network.parameters() if p.requires_grad),
        )

        # Scheduler

    def store(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, epsilon):
        """Select an action using epsilon greedy policy.

        Epsilon chance to take random action.

        Otherwise run the state through the Q network and take the action with the
        highest value.

        Args:
            state (int): current state, one hot encoded
            epsilon (float): exploration rate

        Returns:
            action (int): action to take
        """
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        else:
            # One-hot encode the state
            state_one_hot = np.zeros(self.state_size)
            state_one_hot[state] = 1

            # Convert to tensor and add batch dim
            state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)

            # Get values from Q network
            with torch.no_grad():
                action_values = self.q_network(state_tensor)
            return torch.argmax(action_values).item()

    def train(self, num_step):
        """Train the Q and target network using a batch of experiences from memory.

        Logic for memory accumulation is handled outside of this loop.

        The q values for the current state (from q net) should propagate towards the
        q values for the next state (from target network) discounted by gamma and + the
        immediate reward. Bellman equation.

        Args:
            num_step (int): number of steps taken so far
        """
        # Exit if we don't have enough memories
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.LongTensor(next_states)
        dones = torch.IntTensor(dones)

        # Get Q values (expected return for each state-action pair) for the current state
        states_one_hot_encoded = F.one_hot(states, num_classes=self.state_size).float()
        current_q_values = self.q_network(states_one_hot_encoded)

        # Get Q values for the actions taken.
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values using Bellman equation
        next_states_one_hot_encoded = F.one_hot(
            next_states, num_classes=self.state_size
        ).float()
        next_q_values = self.target_network(next_states_one_hot_encoded)
        next_best_q_value = torch.max(next_q_values, dim=1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_best_q_value

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values.detach())

        # Back propagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the weights of the target network
        if num_step % self.target_q_net_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss

    def save_model(self, path):
        """Save the model weights to a file."""
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        """Load the model weights from a file."""
        self.q_network.load_state_dict(torch.load(path))

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PPOAgent:
    """Proximal Policy Optimization Agent.
    
    PPO is an improvement upon policy gradient methods that incorporates ideas from trust
    region optimization (don't change too much at each step) and clipped surrogate
    objectives (proxy objective fns that are clipped to prevent the agent from straying
    too far with each step).
    """
    def __init__(self, state_size: int, action_size: int, epsilon: float = 0.2, gamma: float = 0.99):
        """Initialize the agent.
        
        Args:
            state_size (int): Number of possible states.
            action_size (int): Number of possible actions.
            epsilon (float, optional): Clipping parameter. Defaults to 0.2.
            gamma (float, optional): Discount factor. Defaults to 0.99.
        """
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters())
        self.optimizer_value = optim.Adam(self.value_network.parameters())
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = []
    
    def store_memory(self, state: torch.Tensor, action: int, reward: int, next_state: torch.Tensor, episode_done: int, old_prob: torch.Tensor):
        """Store an experience in memory.
        
        Args:
            state (torch.Tensor): One hot encoded tensor representing the current state.
            action (int): The action taken.
            reward (int): The reward received.
            next_state (torch.Tensor): One hot encoded tensor representing the next
                state.
            episode_done (int): Whether or not the episode is done.
            old_prob (torch.Tensor): The probability of taking the action in the current
                state.
        """
        self.memory.append((state, action, reward, next_state, episode_done, old_prob))

    def select_action(self, state: torch.Tensor):
        """Select an action using the policy network.
        
        Runs one hot encoded state through the policy network and gets a probability
        distribution over actions.
        Args:
            state (torch.Tensor): One hot encoded tensor representing the current state.

        Returns:
            int: The action to take.
        """
        with torch.no_grad():
            probs = self.policy_network(state)
            # Create a torch distribution that we can sample according to the probabilities
            m = torch.distributions.Categorical(probs)
            selected_action = m.sample().item()
            return selected_action, probs[selected_action] 

    def calculate_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
        return advantages

    def step(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, old_probs: torch.Tensor):
        """Update the policy and value networks using the PPO loss function.

        Args:
            states (torch.Tensor): Batch of one-hot encoded tensors representing the current states.
            actions (torch.Tensor): Batch of actions taken.
            rewards (torch.Tensor): Batch of rewards received.
            next_states (torch.Tensor): Batch of one-hot encoded tensors representing the next states.
            dones (torch.Tensor): Batch of binary flags indicating whether the episode is done.
            old_probs (torch.Tensor): Batch of probabilities of taking the action in the current states.
        """
        for state, action, reward, next_state, episode_done, old_prob in zip(states, actions, rewards, next_states, dones, old_probs):
            # Compute advantage with TD error. How much better is this action compared to
            # the "average" action in the state. In practice, I think PPO uses something
            # called GAE. Keeping it simple for now. Advantage measures how much better
            # an action is compared to the value function estimate. Implicitly,
            # self.value_network(state) estimates the expected return across all actions,
            # similar to "averaging".
            td_target = reward + (1 - episode_done) * self.value_network(next_state)
            advantage = td_target - self.value_network(state)

            # Compute policy loss
            # Existence of advantage in the loss fn: Encourage the policy to increase
            # the probability of actions with positive advantage (good actions) = more
            # loss values, and decrease the probability of actions with negative
            # advantage (bad actions) = more positive loss values.
            # The use of ratio/clamped ratio scales the loss value and acts to create a
            # "trust region". Trust regions stabilize training and work by trust regions
            # magnifying changes that align with past policy changes and reducing changes
            # that do not align.
            # Consider the following scenarios to understand trust regions:
            # 1. If advantage is positive (good action) then the policy loss will be negative.
            # If the ratio is high (network has made this action more likely) then we want to 
            # scale the loss value to be more negative to keep the network moving in 
            # this direction. In a "trusted" direction.
            # 2. If advantage is positive (good action) and the ratio is low
            # (network has made this action less likely), then we want to scale the loss
            # value "up" (less negative) to slow down the network's movement in this
            # direction. In an "untrusted" direction.
            prob = self.policy_network(state)[action]
            ratio = prob / old_prob
            clamped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = - advantage * torch.min(ratio, clamped_ratio)
            
            # WIP up to here
            # Compute value loss
            value_loss = (reward + (1 - episode_done) * self.value_network(next_state) - self.value_network(torch.FloatTensor(state)))**2
            
        # Update policy and value networks
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
