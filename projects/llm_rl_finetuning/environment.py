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

    def compute_classifier_loss(self, sequences: List[str]) -> torch.Tensor:
        """Calculate the loss for a list of sequences based on the AI classifier.

        Args:
            sequences: The list of sequences of actions.

        Returns:
            torch.Tensor: The loss for the sequences.
        """
        losses = []
        for sequence in sequences:
            ai_classifier_output = self.ai_classifier(sequence)
            if ai_classifier_output[0]['label'] == 'Real':
                loss = torch.tensor([1 - ai_classifier_output[0]['score']], device=self.device)
            else:
                loss = torch.tensor([ai_classifier_output[0]['score']], device=self.device)
            losses.append(loss)
        return torch.stack(losses)

