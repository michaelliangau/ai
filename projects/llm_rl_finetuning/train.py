from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import agent
import datasets
import environment
import random
import IPython
from torch.utils.data import DataLoader

# Set a seed for the random number generator to ensure reproducibility
random.seed(0)
torch.autograd.set_detect_anomaly(True)

import sys
sys.path.append("../..")
import common.utils as common_utils

# Hyperparameters
epochs = 10
max_seq_length = 100
learning_rate = 1e-4
device = "cpu"
eval_steps = 100
save_steps = 500
eval = False
batch_size = 2

# Create outputs folder
common_utils.create_folder("outputs")

# Start wandb logging
common_utils.start_wandb_logging(project_name="llm_rl_finetuning")

# Initialize environment and agent
torch_device = common_utils.get_device(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(tokenizer=tokenizer, max_seq_length=max_seq_length, device=torch_device)
simple_agent = agent.SimpleAgent(model=model, tokenizer=tokenizer, learning_rate=learning_rate)

# Set EOS token as pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Move model components to device
model.to(torch_device)

# Dataset
dataset = datasets.load_dataset("alistvt/coqa-stories")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

# Train loop
for epoch in range(epochs):
    for step, batch in enumerate(tqdm(train_dataloader)):
        # # Evaluation step every X steps
        # if eval is True and step % eval_steps == 0:
        #     print("Evaluation Step")
        #     rewards = []
        #     with torch.no_grad():
        #         for prompt in tqdm(eval_dataset, total=len(eval_dataset)):
        #             # Feed the text into the AI classifier
        #             prompt_tensor = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)

        #             _, _, output_decoded = simple_agent.generate_sequence(input_tensor=prompt_tensor, iterations=env.max_seq_length)
        #             reward = env.get_reward(output_decoded)

        #             rewards.append(reward)
            
        #     # Convert rewards list of tensors into a single tensor
        #     rewards_tensor = torch.stack(rewards)
        #     # Calculate mean reward
        #     mean_reward = torch.mean(rewards_tensor).item()
        #     print(f"Mean reward: {mean_reward}")
            
        #     # Log mean reward to wandb
        #     common_utils.log_wandb({"mean_reward": mean_reward})

        # Tokenize text
        text = batch['text']
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(torch_device)

        # Teacher forced forward pass (for MLM loss)
        output, log_probs, output_decoded = simple_agent.forward(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        print(output_decoded)

        reward = env.get_reward(output_decoded)
        # Backfill rewards (terminal reward at end of sequence)
        rewards = reward.repeat(env.max_seq_length)
        
        # TODO Student forward pass (for classifier loss)

        # Compute loss and update policy
        loss = simple_agent.compute_loss(log_probs, rewards)
        simple_agent.optimizer.zero_grad()
        loss.backward()
        simple_agent.optimizer.step()

        # Log loss
        common_utils.log_wandb({"epoch": epoch, "loss": loss})

        if step % save_steps == 0 and step != 0:
            # Save model checkpoint
            torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_{step}.pt')

    print(f'Epoch {epoch}: Loss {loss.item()}')
    # Save model at the end of every epoch
    torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_final.pt')
    
common_utils.end_wandb_logging()
