import torch
from typing import Tuple
from transformers import pipeline
import IPython
from transformers import GPT2Tokenizer

class Environment:
    """This class represents the environment in which the RL agent operates.
    
    TODO This is currently an MVP that returns an arbitrary reward based on arbitrary actions.
    We need to make it meaningfully alter the LLM to get closer to a target sequence.
    """
    def __init__(self, tokenizer: GPT2Tokenizer, max_seq_length: int):
        """Initialize the environment.

        Args:
            tokenizer: The tokenizer to be used by the environment.
            max_seq_length (int): The maximum length of the sequence that can be generated.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.generated_sequence = torch.tensor([])
        self.ai_classifier = pipeline("text-classification", model="roberta-base-openai-detector")

    def step(self, action: int) -> float:
        """Perform a step in the environment with the given action.

        Args:
            action: The action to be performed.

        Returns:
            float: The reward for the action.
        """
        self.generated_sequence = torch.cat((self.generated_sequence, torch.tensor([action])))
        text = self.tokenizer.decode(self.generated_sequence)
        ai_classifier_output = self.ai_classifier(text)

        if ai_classifier_output[0]['label'] == 'Fake':
            reward = 1 - ai_classifier_output[0]['score']
            return reward
        elif ai_classifier_output[0]['label'] == 'Real':
            reward = ai_classifier_output[0]['score']
            return reward

    def _get_reward(self) -> int:
        """Calculate the reward for the current sequence.

        Returns:
            int: The reward for the current sequence.
        """
        # TODO: Calculate a more meaningful reward.
        reward = 0
        return reward

    def reset(self) -> torch.Tensor:
        """Reset the environment to its initial state."""
        self.generated_sequence = torch.tensor([])
        return self.generated_sequence



