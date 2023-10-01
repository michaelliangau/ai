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
do_eval = True
batch_size = 2

# Create outputs folder
common_utils.create_folder("outputs")

# Start wandb logging
# common_utils.start_wandb_logging(project_name="llm_rl_finetuning")

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

# Preprocess data
train_dataset = train_dataset.map(lambda examples: utils.preprocess_data(examples, tokenizer, max_seq_length), batched=True, batch_size=1, num_proc=8, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(lambda examples: utils.preprocess_data(examples, tokenizer, max_seq_length), batched=True, batch_size=1, num_proc=8, remove_columns=eval_dataset.column_names)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)

# Train loop
for epoch in range(epochs):
    for step, batch in enumerate(tqdm(train_dataloader)):
        # TODO: Evaluation step
    
        # Define variables
        input_values = batch['input_values'].to(torch_device)
        labels = batch['labels'].to(torch_device)

        # Forward pass for current token
        action, log_probs = simple_agent.forward(input_values=input_values)
        
        # TODO: Compute NLL loss against target
        IPython.embed()


        # Compute AI classifier loss
        # TODO: Classifier loss is WRONG. You have to make the model generate something new.
        classifier_loss = env.compute_classifier_loss(output_decoded)
        mean_classifier_loss = torch.mean(classifier_loss)

        # Compute total loss
        loss = nll_loss + mean_classifier_loss

        # Backward pass
        simple_agent.optimizer.zero_grad()
        loss.backward()
        simple_agent.optimizer.step()

        # Accumulate loss and increment token counter
        total_nll_loss += nll_loss.item()
        total_classifier_loss += mean_classifier_loss.item()
        num_tokens += 1      

        # Calculate mean loss across the entire sample
        mean_nll_loss = total_nll_loss / num_tokens
        mean_classifier_loss = total_classifier_loss / num_tokens

        # Calculate what % of the total loss is nll or mean classifier loss
        total_loss = mean_nll_loss + mean_classifier_loss
        classifier_loss_percentage = (mean_classifier_loss / total_loss) * 100

        # Log the losses, their percentages, and the epoch loss to wandb
        common_utils.log_wandb({"mean_nll_loss": mean_nll_loss, "mean_classifier_loss": mean_classifier_loss, "classifier_loss_percentage": classifier_loss_percentage, "epoch": epoch, "total_loss": total_loss})

        if step % save_steps == 0 and step != 0:
            # Save model checkpoint
            torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_{step}.pt')

    print(f'Epoch {epoch}: Loss {loss.item()}')
    # Save model at the end of every epoch
    torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_final.pt')
