import numpy as np

class QAgent:
    """Simple agent that implements the Q-learning algorithm."""
    def __init__(self, num_states=16, num_actions=4, alpha=0.9, gamma=0.95, epsilon=0.5):
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
            return np.random.choice(self.num_actions) # explore
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
