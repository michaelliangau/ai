import torch
from typing import Tuple, List
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

    def get_rewards(self, sequences: List[str]) -> torch.Tensor:
        """Calculate the rewards for a list of sequences.

        Args:
            sequences: The list of sequences of actions.

        Returns:
            torch.Tensor: The rewards for the sequences.
        """
        rewards = []
        for sequence in sequences:
            ai_classifier_output = self.ai_classifier(sequence)
            if ai_classifier_output[0]['label'] == 'Fake':
                reward = torch.tensor([1 - ai_classifier_output[0]['score']], device=self.device)
                rewards.append(reward)
            elif ai_classifier_output[0]['label'] == 'Real':
                reward = torch.tensor([ai_classifier_output[0]['score']], device=self.device)
                rewards.append(reward)
        return torch.stack(rewards)

