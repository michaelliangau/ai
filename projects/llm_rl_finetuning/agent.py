import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import IPython

class PPOAgent:
    """Class representing a Proximal Policy Optimization (PPO) agent."""
    def __init__(self, model, tokenizer, learning_rate=1e-4):
        """Initialize the PPOAgent.
        
        Args:
            model: The model to be used by the agent.
            tokenizer: The tokenizer to be used by the agent.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, sequence):
        """Select an action based on the current sequence.

        Args:
            sequence: The current sequence.

        Returns:
            tuple: The selected action and the log probability of the action.
        """
        input_ids = self.tokenizer.encode(sequence)
        input_tensor = torch.tensor([input_ids], dtype=torch.long) # Shape [batch, seq_length]
        
        with torch.no_grad():
            logits = self.model(input_ids=input_tensor).logits
            probs = F.softmax(logits[:, -1, :], dim=-1) # Softmax logits
            m = Categorical(probs) # Converts this into a categorial distribution that can be sampled
            action = m.sample() # Sample from the categorical distribution
            return action.item(), m.log_prob(action)

    def compute_loss(self, log_probs, rewards):
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

