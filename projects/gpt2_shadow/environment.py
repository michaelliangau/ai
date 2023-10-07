import torch
from typing import List
import transformers
import IPython
from sentence_transformers import SentenceTransformer

class Environment:
    """This class represents the environment in which the RL agent operates."""
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, device: str = 'cuda'):
        """Initialize the environment.

        Args:
            tokenizer: The tokenizer to be used by the environment.
            device (str, optional): The device to be used for computations. Defaults to 'cuda'.
        """
        self.tokenizer = tokenizer
        self.ai_classifier = transformers.pipeline("text-classification", model="roberta-base-openai-detector", device=device)
        self.sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.device = device

    def compute_classifier_loss(self, sequences: List[str]) -> torch.Tensor:
        """Calculate the loss for a list of sequences based on the AI classifier.

        Depreciated: This is used in the supervised learning paradigm.

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

    def compute_rl_reward(self, model_output: List[str]) -> torch.Tensor:
        """Calculate the reward for a list of model_output based on the AI classifier.

        This function is used only in the RL paradigm where 'Real' is considered
        good and results in a higher reward.

        # TODO: We're feeding in the prompt from the start, shouldn't do this.

        Args:
            model_output: The list of model_output of actions.

        Returns:
            torch.Tensor: The reward for the model_output.
        """
        rewards = []
        IPython.embed()
        # TODO: WIP Need to bring in the answer in as well and then run a sentence embedding model across the tokens up to the correct index then compute similarity.
        embeddings = self.sentence_embedding_model.encode(model_output)

        for sample in model_output:
            ai_classifier_output = self.ai_classifier(sample)
            if ai_classifier_output[0]['label'] == 'Real':
                reward = torch.tensor(ai_classifier_output[0]['score'], 
                                      device=self.device)
            else:
                reward = torch.tensor(1 - ai_classifier_output[0]['score'], 
                                      device=self.device)
            rewards.append(reward)
        return torch.stack(rewards)

