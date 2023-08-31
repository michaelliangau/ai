from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import agent
import environment
import IPython

# Hyperparameters
epochs = 10
max_seq_length = 100

# Initialize environment and agent
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(max_seq_length=max_seq_length)
ppo_agent = agent.PPOAgent(model, tokenizer)

# Train loop
for epoch in range(epochs):
    generated_sequence = env.reset()
    log_probs = []
    rewards = []

    # Generate sequence
    for _ in range(env.max_seq_length):
        IPython.embed()
        action, log_prob = ppo_agent.select_action(generated_sequence)
        reward, done = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        generated_sequence = torch.cat((generated_sequence, torch.tensor([action])))
        if done:
            break

    # Compute loss and update policy
    loss = ppo_agent.compute_loss(log_probs, rewards)
    ppo_agent.optimizer.zero_grad()
    loss.backward()
    ppo_agent.optimizer.step()
    print(f'Epoch {epoch}: Loss {loss.item()}')