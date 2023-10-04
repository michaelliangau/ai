import torch
from typing import List, Tuple
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import IPython
from transformers import PreTrainedModel, PreTrainedTokenizer
import random
class SimpleAgent:
    """Class representing a simple RL agent."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Initialize the SimpleAgent.
        
        Args:
            model: The model to be used by the agent.
            tokenizer: The tokenizer to be used by the agent.
        """
        self.model = model
        self.tokenizer = tokenizer

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Tokenize a sequence using the agent's tokenizer.

        Args:
            sequence (str): The sequence to be tokenized.

        Returns:
            Tensor[int]: The tokenized sequence as a tensor of integers.
        """
        return torch.tensor(self.tokenizer.encode(sequence))

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

        Used in supervised training regime.

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

    def get_action_and_log_prob_rl(self, state_encoded: torch.Tensor, epsilon: float = 0.1) -> Tuple[int, torch.Tensor]:
        """Get action and log probability for reinforcement learning.

        Args:
            state_encoded (torch.Tensor): Encoded state.
            epsilon (float, optional): Epsilon for epsilon-greedy exploration. 
                Defaults to 0.1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Selected action and its log probability.
        """
        logits = self.model(input_ids=state_encoded).logits
        probs = torch.softmax(logits, dim=-1)
        
        # Epsilon-greedy exploration strategy
        if random.random() < epsilon:
            action = torch.randint(0, logits.shape[-1], (1,))
        else:
            dist = Categorical(probs)
            action = dist.sample()
            action = action[-1:]

        # Log probability for training
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        log_prob = log_prob[-1]
        
        return action, log_prob

    def forward_single(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates the next token based on a given input tensor.

        Args:
            input_values (torch.Tensor): The input tensor to be used for sequence generation.
            attention_mask (torch.Tensor): The attention mask for the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The generated token represented as a tensor, and the logits represented as a tensor.
        """
        action, logits = self.select_action(input_values, attention_mask)
        return action, logits

    def forward_autoregressive(self, input_values: torch.Tensor, attention_mask: torch.Tensor, num_actions: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates multiple tokens autoregressively based on a given input tensor.

        Args:
            input_values (torch.Tensor): The input tensor to be used for sequence generation.
            attention_mask (torch.Tensor): The attention mask for the input tensor.
            num_actions (int): The number of actions to be generated. Default 100.

        Returns:
            torch.Tensor: The generated tokens represented as a tensor.
        """
        actions = []
        for _ in tqdm(range(num_actions)):
            action, _ = self.select_action(input_values, attention_mask)
            actions.append(action)
            input_values = torch.cat((input_values, action.unsqueeze(-1)), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(action).unsqueeze(-1)), dim=-1)
        return torch.stack(actions)
    
    def compute_loss_ppo_rl(states, rewards, log_probs, gamma=0.99, epsilon=0.2):
        
        IPython.embed() # TODO: Build this function
        R = 0
        discounted_rewards = []
        
        # Compute discounted rewards
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        # Update policy by PPO
        log_probs = torch.stack(log_probs)
        discounted_rewards = torch.Tensor(discounted_rewards)
        advantages = discounted_rewards - log_probs.exp()
        
        ratio = (log_probs - log_probs.detach()).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        
        loss = -torch.min(surr1, surr2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

