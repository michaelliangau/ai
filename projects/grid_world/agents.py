import numpy as np
import IPython


class QLearningAgent:
    """Simple agent that implements the Q-learning algorithm."""

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
    def __init__(self, num_states, goal_state_idx, num_actions, theta=1e-8, gamma=0.95):
        """Initialize the ValueIterationAgent.

        Value iteration is a model based RL method

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
    def __init__(self, num_states, goal_state_idx, num_actions, theta=1e-8, gamma=0.95):
        """Initialize the PolicyIterationAgent.

        Policy iteration is a model based RL method

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

        Policy evaluation updates the value function based on the current policy until
        the value function estimate stabilizes.
        Policy improvement then updates the policy based on the current value function.
        This generates a new policy that is guaranteed to be better than or as good as
        the old policy.

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
                # Value of the state is the expected return from the current action under the current policy
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
