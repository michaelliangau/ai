import torch
from typing import List, Tuple, Optional
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import IPython
import transformers
import random
from torch import nn

class ValueNetwork(nn.Module):
    """Class representing a value network for GPT2."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        """Initialize the ValueNetwork.
        
        Args:
            input_dim (int): The input dimension for the value network.
            hidden_dim (int): The hidden dimension for the value network.
            output_dim (int): The output dimension for the value network. Default 1.
        """
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Compute the value estimate for a given state.

        Args:
            input_values (torch.Tensor): The current state represented as a tokenized tensor.

        Returns:
            torch.Tensor: The value estimate for the current state.
        """
        value_estimate = self.value_head(input_values)
        return value_estimate

class SimpleAgent(transformers.GPT2LMHeadModel):
    """Class representing a simple RL agent."""
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        """Initialize the SimpleAgent.
        
        Args:
            model: The model to be used by the agent.
            tokenizer: The tokenizer to be used by the agent.
            value_network: The value network to be used by the agent.
            device: The device to be used by the agent.
        """
        super().__init__(config=model.config)
        self.model = model  
        self.tokenizer = tokenizer
        self.value_head = ValueNetwork(input_dim=model.config.hidden_size, hidden_dim=256, output_dim=1)

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

    def get_action_and_log_prob_rl(self, state: torch.Tensor, action: Optional[torch.Tensor] = None, epsilon: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability for a specific action-state pair.

        Args:
            state (torch.Tensor): Encoded state.
            action (Optional[torch.Tensor]): The specific action to be taken. If None, an action is selected based on epsilon-greedy exploration.
            epsilon (float, optional): Epsilon for epsilon-greedy exploration. Defaults to 0.1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Selected action and its log probability.
        """
        outputs, _ = self.forward(input_values=state)
        logits = outputs.logits[:, -1, :] # Shape: (batch_size, sequence_length, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        
        # Epsilon-greedy exploration strategy
        if action is None:
            if random.random() < epsilon:
                action = torch.randint(0, logits.shape[-1], (1,))
            else:
                dist = Categorical(probs)
                action = dist.sample()
                action = action[-1:]

        # Log probability for training
        dist = Categorical(probs)
        log_prob = dist.log_prob(torch.tensor([action]))
        
        return action, log_prob
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Get the output of the value head.

        Args:
            input_values (torch.Tensor): The input tensor to be used for sequence generation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output of the model and the output of the value head.
        """
        outputs = self.model(input_ids=input_values, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)
        value_pred = self.value_head(last_hidden_state[:, -1, :]).squeeze()  # Shape: (batch_size, sequence_length, 1)
        return outputs, value_pred
    
    def compute_loss_ppo_rl(self, states: List[torch.Tensor], actions: List[torch.Tensor], rewards: List[float], old_log_probs: List[torch.Tensor], gamma: float = 0.99, epsilon: float = 0.2) -> None:
        """Computes the loss for Proximal Policy Optimization (PPO) reinforcement learning.

        Args:
            states (List[torch.Tensor]): List of states.
            actions (List[torch.Tensor]): List of actions.
            rewards (List[float]): List of rewards.
            old_log_probs (List[torch.Tensor]): List of log probabilities from the old policy.
            gamma (float, optional): Discount factor for future rewards. Default is 0.99.
            epsilon (float, optional): Clipping parameter for PPO. Default is 0.2.
        """
        # TODO: I don't think this works with batching right now.
        # Compute expected return in each state. Return = discounted future reward
        R = 0
        discounted_rewards = []
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        # Fix tensors
        old_log_probs = torch.stack(old_log_probs).squeeze()
        discounted_rewards = torch.Tensor(discounted_rewards)

        # Compute value estimates for each state

        advantages = []
        for idx, state in enumerate(states):
            # TODO: Kill the for loop, use an attention mask to make this faster
            _, value_pred = self.forward(input_values=state)
            advantage = discounted_rewards[idx] - value_pred.detach()
            advantages.append(advantage)
        advantages = torch.stack(advantages).squeeze()
        
        # Compute new log probabilities for each state-action pair
        # This should be the same as old_log_probs if loss is computed right after episode
        # finish for the most recent episode.
        new_log_probs = []
        for state, action in zip(states, actions):
            _, log_prob = self.get_action_and_log_prob_rl(state=state, action=action)
            new_log_probs.append(log_prob)
        new_log_probs = torch.stack(new_log_probs).squeeze()
        IPython.embed()
        
        ratio = (new_log_probs - old_log_probs.detach()).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        
        loss = -torch.min(surr1, surr2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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