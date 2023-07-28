import numpy as np

class GridWorld:
    def __init__(self, grid_size=4, hole_count=4):
        """Initialize the environment.

        This is a static grid world environment.
        
        Args:
            grid_size (int): size of the grid (default is 4)
            hole_count (int): number of holes in the grid (default is 4)
        """
        self.grid_size = grid_size
        self.state_space = np.arange(grid_size * grid_size)
        self.action_space = np.arange(4) # 0: Up, 1: Right, 2: Down, 3: Left

        # Initialize the grid
        self.hole_count = hole_count
        self.state = 0
        self.grid = np.full((self.grid_size, self.grid_size), 'F') # Fill the grid with frozen blocks
        self.grid[0, 0] = 'S' # Place the start block
        self.grid[-1, -1] = 'G' # Place the goal block

        # Place the hole blocks randomly
        for _ in range(self.hole_count):
            row, col = np.random.randint(self.grid_size, size=2)
            # Ensure we don't place a hole at the start or goal
            while (row, col) in [(0, 0), (self.grid_size - 1, self.grid_size - 1)]:
                row, col = np.random.randint(self.grid_size, size=2)
            self.grid[row, col] = 'H'

    def _get_grid_position(self, state):
        """Converts a state number to a position in a 4x4 grid.

        Args:
            state (int): The state number, between 0 and 15 inclusive.

        Returns:
            tuple: The grid position as (row, column).
        """
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def reset(self):
        """Reset the environment to the initial state

        The grid stays the same, only the agent's state is reset to the start block.

        Returns:
            state (int): initial state
        """
        self.state = 0
        return self.state


    # Step 3: Take an action and return the next state and reward
    def step(self, action):
        """Take an action and return the next state and reward

        Args:
            action (int): action to take

        Returns:
            state (int): next state
            reward (int): reward
        """
        row, col = self._get_grid_position(self.state)
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Right
            col = min(col + 1, self.grid_size - 1)
        elif action == 2:  # Down
            row = min(row + 1, self.grid_size - 1)
        elif action == 3:  # Left
            col = max(col - 1, 0)
            
        # Update the state
        next_state = row * self.grid_size + col
        self.state = next_state
        
        # Get reward based on current state
        reward = self._get_reward(row, col)

        return next_state, reward

    def _get_reward(self, row, col):
        """Return the reward based on the current state.

        Args:
            row (int): row of the current state
            col (int): column of the current state
        
        Returns:
            reward (int): -1 for falling into a hole, 1 for reaching the goal, and 0 otherwise
        """
        if self.grid[row, col] == 'H':  # If the agent falls into a hole
            return -1
        elif self.grid[row, col] == 'G':  # If the agent reaches the goal
            return 1
        else:  # If the agent is on a frozen block
            return 0
        
    def transitions(self, state, action):
        """Compute the transition for the given state and action.

        Transition probabiltiies are used in value iteration algorithms. Transition
        probs are 1.0 here because GridWorld is determinstic. This may change for
        different environments.
        
        Args:
            state: The current state.
            action: The action to perform.

        Returns:
            A list of tuples with (probability, next_state, reward, done).
        """
        row, col = self._get_grid_position(state)
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Right
            col = min(col + 1, self.grid_size - 1)
        elif action == 2:  # Down
            row = min(row + 1, self.grid_size - 1)
        elif action == 3:  # Left
            col = max(col - 1, 0)
        
        transition_prob = 1.0
        next_state = row * self.grid_size + col
        reward = self._get_reward(row, col)
        done = (self.grid[row, col] == 'H') or (self.grid[row, col] == 'G')

        return [(transition_prob, next_state, reward, done)]
