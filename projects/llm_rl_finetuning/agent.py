import torch
from typing import List, Tuple
import torch.nn.functional as F
from torch.distributions import Categorical
import IPython
from transformers import PreTrainedModel, PreTrainedTokenizer

class SimpleAgent:
    """Class representing a simple RL agent."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, learning_rate: float = 1e-4):
        """Initialize the SimpleAgent.
        
        Args:
            model: The model to be used by the agent.
            tokenizer: The tokenizer to be used by the agent.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """Tokenize a sequence using the agent's tokenizer.

        Args:
            sequence (str): The sequence to be tokenized.

        Returns:
            Tensor[int]: The tokenized sequence as a tensor of integers.
        """
        return self.tokenizer.encode(sequence)

    def decode_sequence(self, sequence: torch.Tensor) -> List[str]:
        """Decode a sequence using the agent's tokenizer.

        Args:
            sequence (torch.Tensor): The sequence to be decoded. Shape: (batch, len)

        Returns:
            List[str]: The decoded sequences.
        """
        return [self.tokenizer.decode(seq.tolist()) for seq in sequence]
    
    def select_action(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects an action based on the current sequence.

        Args:
            input_values (torch.Tensor): The current sequence represented as a tokenized tensor.
            attention_mask (torch.Tensor): The attention mask for the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The selected action represented as a tensor, and the log probability of the action represented as a tensor.
        """
        logits = self.model(input_ids=input_values, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1) # Log softmax logits
        m = Categorical(logits=log_probs.exp()) # Converts this into a categorial distribution that can be sampled
        action = m.sample() # Sample from the categorical distribution, this is where LM stochasticity comes from.
        return action, logits

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the next token based on a given input tensor.

        Args:
            input_values (torch.Tensor): The input tensor to be used for sequence generation.
            attention_mask (torch.Tensor): The attention mask for the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The generated token represented as a tensor, its log probability represented as a tensor, and the logits represented as a tensor.
        """
        action, logits = self.select_action(input_values, attention_mask)
        return action, logits


