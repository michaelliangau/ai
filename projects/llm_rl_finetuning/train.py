from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datasets
import torch
import agent
import environment
import IPython

import sys
sys.path.append("../..")
import common.utils as common_utils

# Hyperparameters
epochs = 10
max_seq_length = 100
learning_rate = 1e-4
device = "cpu"

# Start wandb logging
common_utils.start_wandb_logging(project_name="llm_rl_finetuning")

# Initialize environment and agent
torch_device = common_utils.get_device(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(tokenizer=tokenizer, max_seq_length=max_seq_length, device=torch_device)
ppo_agent = agent.SimpleAgent(model=model, tokenizer=tokenizer, learning_rate=learning_rate)

# Move model components to device
model.to(torch_device)

# Dataset
huggingface_dataset = datasets.load_dataset('squad')
questions = []
for example in huggingface_dataset['train']:
    question = f"{example['question']}\n"
    questions.append(question)

# Train loop
for epoch in range(epochs):
    for step, question in tqdm(enumerate((questions)), total=len(questions)):
        env.reset()
        full_sequence = tokenizer.encode(question, return_tensors='pt').to(torch_device)
        log_probs = []

        # Generate sequence
        for _ in range(env.max_seq_length):
            # Produce a token
            action, log_prob = ppo_agent.select_action(full_sequence)
            log_probs.append(log_prob)
            
            # Get reward from environment
            reward = env.step(action)

            # Add action to full sequence
            full_sequence = torch.cat((full_sequence, torch.tensor([[action]]).to(torch_device)), dim=-1)
        
        # Backfill rewards (terminal reward at end of sequence)
        rewards = [reward] * env.max_seq_length
        
        # Compute loss and update policy
        loss = ppo_agent.compute_loss(log_probs, rewards)
        ppo_agent.optimizer.zero_grad()
        loss.backward()
        ppo_agent.optimizer.step()

        # Log loss
        common_utils.log_wandb({"epoch": epoch, "loss": loss})
        print(f'Loss {loss.item()}')
    print(f'Epoch {epoch}: Loss {loss.item()}')

        # TODO: Eval every 100 steps
common_utils.end_wandb_logging()
