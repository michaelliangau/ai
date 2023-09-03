import torch
from typing import Tuple
import transformers
import IPython

class Environment:
    """This class represents the environment in which the RL agent operates."""
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, max_seq_length: int, device: str = 'cuda'):
        """Initialize the environment.

        Args:
            tokenizer: The tokenizer to be used by the environment.
            max_seq_length (int): The maximum length of the sequence that can be generated.
            device (str, optional): The device to be used for computations. Defaults to 'cuda'.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = torch.device(device)
        self.generated_sequence = torch.tensor([]).to(self.device)
        self.ai_classifier = transformers.pipeline("text-classification", model="roberta-base-openai-detector")

    def step(self, action: int) -> float:
        """Perform a step in the environment with the given action.

        Args:
            action: The action to be performed.

        Returns:
            float: The reward for the action.
        """
        self.generated_sequence = torch.cat((self.generated_sequence, torch.tensor([action])))
        text = self.tokenizer.decode(self.generated_sequence)
        ai_classifier_output = self.ai_classifier(text)

        if ai_classifier_output[0]['label'] == 'Fake':
            reward = 1 - ai_classifier_output[0]['score']
            return reward
        elif ai_classifier_output[0]['label'] == 'Real':
            reward = ai_classifier_output[0]['score']
            return reward

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



