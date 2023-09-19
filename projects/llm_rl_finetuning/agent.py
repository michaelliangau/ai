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

    def decode_sequence(self, sequence: List[int]) -> str:
        """Decode a sequence using the agent's tokenizer.

        Args:
            sequence (List[int]): The sequence to be decoded.

        Returns:
            str: The decoded sequence.
        """
        return self.tokenizer.decode(sequence)
    
    def select_action(self, input_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Select an action based on the current sequence.

        Args:
            input_tensor: The current sequence as a tokenized tensor.

        Returns:
            tuple: The selected action and the log probability of the action.
        """
        logits = self.model(input_ids=input_tensor).logits
        probs = F.softmax(logits[:, -1, :], dim=-1) # Softmax logits
        m = Categorical(probs) # Converts this into a categorial distribution that can be sampled
        action = m.sample() # Sample from the categorical distribution, this is where LM stochasticity comes from.
        return action.item(), m.log_prob(action)

    def generate_sequence(self, input_tensor: torch.Tensor, iterations: int) -> Tuple[torch.Tensor, str]:
        """Generate a sequence based on a given input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor to be used for sequence generation.
            iterations (int): The number of iterations to generate the sequence.

        Returns:
            Tuple[torch.Tensor, str]: The generated sequence as a tensor and as a decoded string.
        """
        # Initialize the sequence with the last token of the input tensor
        sequence = input_tensor
        output_sequence = torch.tensor([[]]).to(input_tensor.device)

        # Generate the sequence
        log_probs = torch.tensor([]).to(input_tensor.device)
        for _ in range(iterations):
            action, log_prob = self.select_action(sequence)
            sequence = torch.cat((sequence, torch.tensor([[action]]).to(input_tensor.device)), dim=-1)
            output_sequence = torch.cat((output_sequence, torch.tensor([[action]]).to(input_tensor.device)), dim=-1)
            log_probs = torch.cat((log_probs, log_prob.unsqueeze(0)), dim=-1)

        # Decode the sequence
        decoded_sequence = self.decode_sequence(output_sequence[0].tolist())

        return output_sequence, log_probs, decoded_sequence

    def compute_loss(self, log_probs: List[torch.Tensor], rewards: List[float]) -> torch.Tensor:
        """Compute the loss based on the log probabilities and rewards.

        I think the way to go about this is to build a mixed CE loss
        (normal LM loss) and then layer on top an adversarial loss that is based on the
        outputs of https://huggingface.co/roberta-base-openai-detector (GPT-2 detector).
        This way we can maintain LM performance while also making it harder for the detector
        to detect it.

        As an MVP lets just do the classifier adversarial loss from rewards. It is
        just the summed total of the terminal reward multiplied by the log_prob of each
        action.

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
        # High reward and low probability = large negative log_prob value * high reward
        # Low reward and high probability = small negative log_prob value * low reward
        # The network will converge towards high probability actions (low scaling of reward)
        # that get chosen again and again. Assuming reward is constant.
        # You can also think of log_probs as scaling the reward value.
        # This line of thinking is different to how we think of supervised training.
        # The main difference being that it's easier to think of each action having
        # it's own loss fn as opposed to the entire network optimizing for a single
        # north star loss value.
        # Ideally would like to get another set of eyes on this.
        policy_loss = -(log_probs * rewards).sum()

        return policy_loss


