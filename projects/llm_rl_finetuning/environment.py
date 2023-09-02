import torch
from typing import Tuple
class Environment:
    """This class represents the environment in which the RL agent operates.
    
    TODO This is currently an MVP that returns an arbitrary reward based on arbitrary actions.
    We need to make it meaningfully alter the LLM to get closer to a target sequence.
    """
    def __init__(self, max_seq_length: int):
        """Initialize the environment.

        Args:
            max_seq_length (int): The maximum length of the sequence that can be generated.
        """
        self.max_seq_length = max_seq_length
        self.generated_sequence = torch.tensor([])

    def step(self, action: int) -> Tuple[int, bool]:
        """Perform a step in the environment with the given action.

        Args:
            action: The action to be performed.

        Returns:
            tuple: A tuple containing the reward and a boolean indicating whether the
                sequence has reached its maximum length.
        """
        # TODO: Can we remove the self.generated_sequence stuff?
        self.generated_sequence = torch.cat((self.generated_sequence, torch.tensor([action])))

        if len(self.generated_sequence) >= self.max_seq_length:
            reward = self._get_reward()
            return reward, True
        
        return 0, False

    def _get_reward(self) -> int:
        """Calculate the reward for the current sequence.

        Returns:
            int: The reward for the current sequence.
        """
        # TODO: Calculate a more meaningful reward.
        reward = 0
        return reward

    def reset(self) -> torch.Tensor:
        """Reset the environment to its initial state."""
        self.generated_sequence = torch.tensor([])
        return self.generated_sequence



