from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datasets
import torch
import agent
import environment
import IPython
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
huggingface_dataset = datasets.load_dataset('squad')
questions = []
for example in huggingface_dataset['train']:
    question = f"{example['question']}\n"
    questions.append(question)
random.shuffle(questions)

# Train loop
for epoch in range(epochs):
    for step, question in tqdm(enumerate((questions)), total=len(questions)):
        env.reset()
        full_sequence = tokenizer.encode(question, return_tensors='pt').to(torch_device)
        log_probs = []

        # Generate sequence
        for _ in range(env.max_seq_length):
            # Produce a token
            action, log_prob = simple_agent.select_action(full_sequence)
            log_probs.append(log_prob)
            
            # Get reward from environment
            reward = env.step(action)

            # Add action to full sequence
            full_sequence = torch.cat((full_sequence, torch.tensor([[action]]).to(torch_device)), dim=-1)
        
        # Backfill rewards (terminal reward at end of sequence)
        rewards = [reward] * env.max_seq_length
        
        # Compute loss and update policy
        loss = simple_agent.compute_loss(log_probs, rewards)
        simple_agent.optimizer.zero_grad()
        loss.backward()
        simple_agent.optimizer.step()

        # Log loss
        common_utils.log_wandb({"epoch": epoch, "loss": loss})

        # Evaluation step every 100 steps
        if step % eval_steps == 0:
            print("Evaluation Step")
            # Use a subset of the squad test set as the benchmark dataset
            indices = random.sample(range(1, 10001), 10)
            # Use a subset of the squad test set as the benchmark dataset
            benchmark_dataset = [huggingface_dataset['validation'][i] for i in indices]
            rewards = []
            for data in benchmark_dataset:
                # Feed the text into the AI classifier
                question = data['question']
                question_tensor = tokenizer.encode(question, return_tensors='pt').to(torch_device)
                model_output = simple_agent.generate_sequence(input_tensor=question_tensor, iterations=env.max_seq_length)
                classifier_output = env.ai_classifier(model_output)
                
                # Calculate reward from classifier output
                if classifier_output[0]['label'] == 'Fake':
                    reward = 1 - classifier_output[0]['score']
                elif classifier_output[0]['label'] == 'Real':
                    reward = classifier_output[0]['score']
                rewards.append(reward)
            
            # Calculate mean reward
            mean_reward = sum(rewards) / len(rewards)
            print(f"Mean reward: {mean_reward}")
            
            # Log mean reward to wandb
            common_utils.log_wandb({"mean_reward": mean_reward})

            # Save model checkpoint
            torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_{step}.pt')

    print(f'Epoch {epoch}: Loss {loss.item()}')
common_utils.end_wandb_logging()
