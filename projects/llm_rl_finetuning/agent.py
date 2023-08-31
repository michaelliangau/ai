import torch
from typing import List, Tuple
import torch.nn.functional as F
from torch.distributions import Categorical
import IPython
from transformers import PreTrainedModel, PreTrainedTokenizer

class PPOAgent:
    """Class representing a Proximal Policy Optimization (PPO) agent."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, learning_rate: float = 1e-4):
        """Initialize the PPOAgent.
        
        Args:
            model: The model to be used by the agent.
            tokenizer: The tokenizer to be used by the agent.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def tokenize_sequence(self, sequence: str) -> torch.Tensor[int]:
        """Tokenize a sequence using the agent's tokenizer.

        Args:
            sequence (str): The sequence to be tokenized.

        Returns:
            Tensor[int]: The tokenized sequence as a tensor of integers.
        """
        return self.tokenizer.encode(sequence)

    def select_action(self, input_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Select an action based on the current sequence.

        Args:
            input_tensor: The current sequence as a tokenized tensor.

        Returns:
            tuple: The selected action and the log probability of the action.
        """
        with torch.no_grad():
            logits = self.model(input_ids=input_tensor).logits
            probs = F.softmax(logits[:, -1, :], dim=-1) # Softmax logits
            m = Categorical(probs) # Converts this into a categorial distribution that can be sampled
            action = m.sample() # Sample from the categorical distribution
            return action.item(), m.log_prob(action)

    def compute_loss(self, log_probs: List[torch.Tensor], rewards: List[float]) -> torch.Tensor:
        """Compute the loss based on the log probabilities and rewards.

        Args:
            log_probs: The log probabilities of the actions taken.
            rewards: The rewards received for the actions taken.

        Returns:
            torch.Tensor: The computed loss.
        """
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R  # Discount factor
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        return torch.stack(policy_loss).sum()


