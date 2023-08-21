class Environment:
    """This class represents the environment in which the RL agent operates."""
    def __init__(self, max_length):
        """Initialize the environment.

        Args:
            max_length (int): The maximum length of the sequence that can be generated.
        """
        self.max_length = max_length
        self.current_index = 0
        self.generated_sequence = []

    def step(self, action):
        """Perform a step in the environment with the given action.

        Args:
            action: The action to be performed.

        Returns:
            tuple: A tuple containing the reward and a boolean indicating whether the
                sequence has reached its maximum length.
        """
        self.generated_sequence.append(action)
        self.current_index += 1

        if self.current_index >= self.max_length:
            reward = self._get_reward()
            return reward, True
        
        return 0, False

    def _get_reward(self):
        """Calculate the reward for the current sequence.

        Returns:
            int: The reward for the current sequence.
        """
        # TODO - Call API with generated sequence
        reward = 0
        return reward

    def reset(self):
        """Reset the environment to its initial state.

        Returns:
            int: The initial state of the environment.
        """
        self.current_index = 0
        self.generated_sequence = []
        return 0

