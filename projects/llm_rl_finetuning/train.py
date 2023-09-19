from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import agent
import environment
import random
# Set a seed for the random number generator to ensure reproducibility
random.seed(0)

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

# Create outputs folder
common_utils.create_folder("outputs")

# Start wandb logging
common_utils.start_wandb_logging(project_name="llm_rl_finetuning")

# Initialize environment and agent
torch_device = common_utils.get_device(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
env = environment.Environment(tokenizer=tokenizer, max_seq_length=max_seq_length, device=torch_device)
simple_agent = agent.SimpleAgent(model=model, tokenizer=tokenizer, learning_rate=learning_rate)

# Move model components to device
model.to(torch_device)

# Dataset
train_dataset = ["Once upon a time in a quiet village, "] * 100
eval_dataset = ["Once upon a time in a quiet village, "] * 50

# Train loop
for epoch in range(epochs):
    for step, prompt in tqdm(enumerate((train_dataset)), total=len(train_dataset)):
        prompt_tensor = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)

        output, log_probs, output_decoded = simple_agent.generate_sequence(input_tensor=prompt_tensor, iterations=env.max_seq_length)
        reward = env.get_reward(output_decoded)
        # Backfill rewards (terminal reward at end of sequence)
        rewards = reward.repeat(env.max_seq_length)
        
        # Compute loss and update policy
        loss = simple_agent.compute_loss(log_probs, rewards)
        simple_agent.optimizer.zero_grad()
        loss.backward()
        simple_agent.optimizer.step()

        # Log loss
        common_utils.log_wandb({"epoch": epoch, "loss": loss})

        # Evaluation step every X steps
        if step % eval_steps == 0:
            print("Evaluation Step")
            rewards = []
            with torch.no_grad():
                for prompt in tqdm(eval_dataset, total=len(eval_dataset)):
                    # Feed the text into the AI classifier
                    prompt_tensor = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)

                    output, log_probs, output_decoded = simple_agent.generate_sequence(input_tensor=prompt_tensor, iterations=env.max_seq_length)
                    reward = env.get_reward(output_decoded)

                    rewards.append(reward)
            
            # Convert rewards list of tensors into a single tensor
            rewards_tensor = torch.stack(rewards)
            # Calculate mean reward
            mean_reward = torch.mean(rewards_tensor).item()
            print(f"Mean reward: {mean_reward}")
            
            # Log mean reward to wandb
            common_utils.log_wandb({"mean_reward": mean_reward})

        if step % save_steps == 0 and step != 0:
            # Save model checkpoint
            torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_{step}.pt')

    print(f'Epoch {epoch}: Loss {loss.item()}')
    # Save model at the end of every epoch
    torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_final.pt')
    
common_utils.end_wandb_logging()
