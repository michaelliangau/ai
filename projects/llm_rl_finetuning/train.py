from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datasets
import torch
import agent
import environment
import IPython

# Hyperparameters
epochs = 10
max_seq_length = 500

# Initialize environment and agent
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(tokenizer=tokenizer, max_seq_length=max_seq_length)
ppo_agent = agent.SimpleAgent(model=model, tokenizer=tokenizer)

# Dataset
huggingface_dataset = datasets.load_dataset('squad')
questions = []
for example in huggingface_dataset['train']:
    question = f"{example['question']}\n"
    questions.append(question)

# Train loop
for epoch in range(epochs):
    for question in questions:
        print(f'Epoch {epoch}: {question}')
        env.reset()
        full_sequence = tokenizer.encode(question, return_tensors='pt')
        log_probs = []
        rewards = []

        # Generate sequence
        for _ in range(env.max_seq_length):
            action, log_prob = ppo_agent.select_action(full_sequence)
            reward = env.step(action)
            
            # TODO do we need this?
            log_probs.append(log_prob)
            rewards.append(reward)

            # Add action to full sequence
            full_sequence = torch.cat((full_sequence, torch.tensor([[action]])), dim=-1)
            print(tokenizer.decode(full_sequence[0]))

        # Compute loss and update policy
        loss = ppo_agent.compute_loss(log_probs, rewards)
        ppo_agent.optimizer.zero_grad()
        loss.backward()
        ppo_agent.optimizer.step()
        print(f'Epoch {epoch}: Loss {loss.item()}')
