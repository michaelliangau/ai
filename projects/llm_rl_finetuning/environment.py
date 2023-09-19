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
        self.ai_classifier = transformers.pipeline("text-classification", model="roberta-base-openai-detector", device=device)
        self.device = device

    def get_reward(self, sequence: str) -> torch.Tensor:
        """Calculate the reward for a given sequence.

        Args:
            sequence: The sequence of actions.

        Returns:
            torch.Tensor: The reward for the sequence.
        """
        # TODO May need a better reward fn, model is hacking it.
        ai_classifier_output = self.ai_classifier(sequence)

        if ai_classifier_output[0]['label'] == 'Fake':
            reward = torch.tensor([1 - ai_classifier_output[0]['score']], device=self.device)
            return reward
        elif ai_classifier_output[0]['label'] == 'Real':
            reward = torch.tensor([ai_classifier_output[0]['score']], device=self.device)
            return reward

