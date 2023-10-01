from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import agent
import datasets
import environment
import utils
import random
import IPython
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
do_eval = True
train_batch_size = 10
eval_batch_size = 10
num_token_generations = 100

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
tokenizer.pad_token = tokenizer.eos_token # <|endoftext|>
tokenizer.pad_token_id = tokenizer.eos_token_id # 50256

# Move model components to device
model.to(torch_device)

# Dataset
dataset = datasets.load_dataset("alistvt/coqa-stories")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

# Preprocess data
train_dataset = train_dataset.map(lambda examples: utils.preprocess_data(examples, tokenizer, max_seq_length), batched=True, batch_size=1, num_proc=8, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(lambda examples: utils.preprocess_data(examples, tokenizer, max_seq_length), batched=True, batch_size=1, num_proc=8, remove_columns=eval_dataset.column_names)
eval_dataset = eval_dataset.select(range(20)) # Small subset for quicker evaluation

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=lambda batch: utils.collate_fn(batch, tokenizer.pad_token_id))
eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True, collate_fn=lambda batch: utils.collate_fn(batch, tokenizer.pad_token_id))

# Train loop
for epoch in range(epochs):
    for step, batch in enumerate(tqdm(train_dataloader)):

        # Evaluate model
        if step % eval_steps == 0 and do_eval:
            print(f'Evaluating at step {step}...')
            eval_loss = 0
            eval_steps_count = 0
            for eval_batch in tqdm(eval_dataloader):
                eval_input_values = eval_batch['input_values'].to(torch_device)
                eval_attention_mask = eval_batch['attention_mask'].to(torch_device)
                eval_actions = simple_agent.forward_autoregressive(input_values=eval_input_values, attention_mask=eval_attention_mask, num_actions=num_token_generations)
                eval_input_values_no_pad = [eval_input_values[i][eval_attention_mask[i] != 0] for i in range(eval_input_values.size(0))]
                eval_actions = eval_actions.transpose(0, 1)
                eval_full_generation = [torch.cat((eval_input_values_no_pad[i], eval_actions[i]), dim=-1) for i in range(len(eval_input_values_no_pad))]
                eval_decoded_sequence = simple_agent.decode_sequence(eval_full_generation)
                eval_classifier_loss = env.compute_classifier_loss(eval_decoded_sequence)
                eval_loss += eval_classifier_loss.mean().item()
                eval_steps_count += 1
            eval_loss /= eval_steps_count
            common_utils.log_wandb({"eval_loss": eval_loss, "epoch": epoch})
    
        # Define variables
        input_values = batch['input_values'].to(torch_device)
        labels = batch['labels'].to(torch_device)
        attention_mask = batch['attention_mask'].to(torch_device)

        # Forward pass for current token
        action, logits = simple_agent.forward_single(input_values=input_values, attention_mask=attention_mask)
        pred = logits[:, -1, :]

        # Compute Cross Entropy loss against target
        ce_loss = torch.nn.functional.cross_entropy(pred, labels.squeeze())

        # Generate 100 tokens from the input_values
        actions = simple_agent.forward_autoregressive(input_values=input_values, attention_mask=attention_mask, num_actions=num_token_generations)
        
        # Decode the actions
        input_values_no_pad = [input_values[i][attention_mask[i] != 0] for i in range(input_values.size(0))]
        actions = actions.transpose(0, 1)
        full_generation = [torch.cat((input_values_no_pad[i], actions[i]), dim=-1) for i in range(len(input_values_no_pad))]
        decoded_sequence = simple_agent.decode_sequence(full_generation)

        # Compute classifier loss
        classifier_loss = env.compute_classifier_loss(decoded_sequence)
        mean_classifier_loss = classifier_loss.mean()

        # Compute total loss
        loss = ce_loss + mean_classifier_loss

        # Backward pass
        simple_agent.optimizer.zero_grad()
        loss.backward()
        simple_agent.optimizer.step()

        classifier_loss_percentage = (mean_classifier_loss / loss) * 100

        # Log the losses, their percentages, and the epoch loss to wandb
        common_utils.log_wandb({"classifier_loss": mean_classifier_loss, "cross_entropy_loss": ce_loss, "classifier_loss_percentage": classifier_loss_percentage, "epoch": epoch, "total_loss": loss})

        if step % save_steps == 0 and step != 0:
            # Save model checkpoint
            torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_{step}.pt')

    print(f'Epoch {epoch}: Loss {loss.item()}')
    # Save model at the end of every epoch
    torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_final.pt')
