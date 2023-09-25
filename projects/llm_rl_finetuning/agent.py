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
    
    def select_action(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an action based on the current sequence.

        Args:
            input_tensor (torch.Tensor): The current sequence as a tokenized tensor.
            attention_mask (torch.Tensor): The attention mask for the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The selected action as a tensor, and the log probability of the action as a tensor.
        """
        logits = self.model(input_ids=input_tensor, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1) # Log softmax logits
        m = Categorical(logits=log_probs.exp()) # Converts this into a categorial distribution that can be sampled
        action = m.sample() # Sample from the categorical distribution, this is where LM stochasticity comes from.
        return action, log_probs

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the next token based on a given input tensor.

        Args:
            input_ids (torch.Tensor): The input tensor to be used for sequence generation.
            attention_mask (torch.Tensor): The attention mask for the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The generated token as a tensor, and its log probability as a tensor.
        """
        action, log_probs = self.select_action(input_ids, attention_mask)
        return action, log_probs

    def compute_loss(self, log_probs: List[torch.Tensor], rewards: List[float]) -> torch.Tensor:
        """Compute the loss based on the log probabilities and rewards.

        I think the way to go about this is to build a mixed CE loss
        (normal LM loss) and then layer on top an adversarial loss that is based on the
        outputs of https://huggingface.co/roberta-base-openai-detector (GPT-2 detector).
        This way we can maintain LM performance while also making it harder for the detector
        to detect it.

        Args:
            log_probs: The log probabilities of the actions taken.
            rewards: The rewards received for the actions taken.

        Returns:
            torch.Tensor: The computed loss.
        """

        # Calculate policy loss
        # Policy loss calculations try to maximise expected return (prob * reward),
        # and we assume reward is a non-controllable factor in this. We can think of
        # it as if loss is categorical to the specific action in question. Each action
        # has its own unique loss value. So we want the network to have higher loss for
        # low probability/high reward action, to make bigger update. Conversely, low
        # reward/high probability actions should have low loss, to relatively make smaller
        # weight update. Numerically:
        # High reward and low probability = large value * high reward = large loss
        # Low reward and high probability = small value * low reward = small loss
        # The network will converge towards high probability actions (low scaling of reward)
        # that get chosen again and again. Assuming reward is constant.
        # You can also think of log_probs as scaling the reward value.
        # This line of thinking is different to how we think of supervised training.
        # The main difference being that it's easier to think of each action having
        # it's own loss fn as opposed to the entire network optimizing for a single
        # north star loss value.
        # Ideally would like to get another set of eyes on this.
        IPython.embed()
        # policy_loss = -(log_probs * rewards).mean()
        policy_loss = 1 - (rewards * log_probs).mean()

        return policy_loss


