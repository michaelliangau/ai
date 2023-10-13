import torch
from typing import List
import transformers
import IPython


class Environment:
    """This class represents the environment in which the RL agent operates."""

    def __init__(self, tokenizer: transformers.GPT2Tokenizer, device: str = "cuda"):
        """Initialize the environment.

        Args:
            tokenizer: The tokenizer to be used by the environment.
            device (str, optional): The device to be used for computations. Defaults to 'cuda'.
        """
        self.tokenizer = tokenizer
        self.ai_classifier = transformers.pipeline(
            "text-classification", model="roberta-base-openai-detector", device=device
        )
        self.device = device
        self.reward_model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                "sugam11/gpt2-rlhf-reward"
            ).to(device)
        )
        self.reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/DialogRPT-updown"
        )

    def compute_rl_classifier_reward(self, model_output: List[str]) -> torch.Tensor:
        """Calculate the reward for a list of model_output based on the AI classifier.

        This function is used only in the RL paradigm where 'Real' is considered
        good and results in a higher reward.

        Args:
            model_output: The list of model_output of actions.

        Returns:
            torch.Tensor: The reward for the model_output.
        """
        rewards = []
        for sample in model_output:
            ai_classifier_output = self.ai_classifier(sample)
            if ai_classifier_output[0]["label"] == "Real":
                reward = torch.tensor(
                    ai_classifier_output[0]["score"], device=self.device
                )
            else:
                reward = torch.tensor(
                    1 - ai_classifier_output[0]["score"], device=self.device
                )
            rewards.append(reward)
        return torch.stack(rewards)

    def compute_rlhf_reward(
        self, model_outputs: List[str], states: List[str]
    ) -> torch.Tensor:
        """Calculate the reward for a list of model_outputs and states based on the AI classifier.

        Args:
            model_outputs: The list of model_outputs of actions.
            states: The list of states.

        Returns:
            torch.Tensor: The reward for the model_outputs and states.
        """
        encoded_inputs = self.reward_tokenizer(model_outputs, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            rewards = self.reward_model(**encoded_inputs).logits
        rewards = rewards.squeeze(-1)
        return rewards

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
            if ai_classifier_output[0]["label"] == "Real":
                loss = torch.tensor(
                    [1 - ai_classifier_output[0]["score"]], device=self.device
                )
            else:
                loss = torch.tensor(
                    [ai_classifier_output[0]["score"]], device=self.device
                )
            losses.append(loss)
        return torch.stack(losses)
