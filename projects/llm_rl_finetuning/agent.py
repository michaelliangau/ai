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

    def select_action(self, state):
        """Select an action based on the current state.

        Args:
            state: The current state.

        Returns:
            tuple: The selected action and the log probability of the action.
        """
        input_ids = torch.tensor([state], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
            probs = F.softmax(logits[:, -1, :], dim=-1)
            m = Categorical(probs)
            action = m.sample()
            return action.item(), m.log_prob(action)

    def train(self, environment, epochs=1000):
        """Train the agent.

        Args:
            environment: The environment in which the agent operates.
            epochs (int, optional): The number of epochs for training. Defaults to 1000.
        """
        max_length = environment.max_length

        for epoch in range(epochs):
            state = environment.reset()
            log_probs = []
            rewards = []

            # Generate sequence
            for t in range(max_length):
                action, log_prob = self.select_action(state)
                reward, done = environment.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                state.append(action)
                if done:
                    break

            # Compute loss and update policy
            loss = self.compute_loss(log_probs, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch}: Loss {loss.item()}')

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

