from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch
import agent
import datasets
import environment
import utils
import random
import IPython
from torch.utils.data import DataLoader
import transformers
torch.autograd.set_detect_anomaly(True)
# Set a seed for the random number generator to ensure reproducibility
random.seed(0)

import sys
sys.path.append("../..")
import common.utils as common_utils

# Hyperparameters
experiment_name = "dev-cuda"
num_episodes = 100
max_seq_length = 50
learning_rate = 4e-3
device = "cuda"
eval_steps = 100
save_steps = 500
do_eval = True
train_batch_size = 24
eval_batch_size = 24
eval_dataset_size = 96
warmup_steps = 100
# TODO: Add episolon gamma, beta from agent

# Create outputs folder
common_utils.create_folder("outputs")

# Start wandb logging
common_utils.start_wandb_logging(project_name="llm_rl_finetuning", name=experiment_name)

# Initialize environment and agent
torch_device = common_utils.get_device(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
env = environment.Environment(tokenizer=tokenizer, device=torch_device)
actor_critic_agent = agent.ActorCriticAgent(tokenizer=tokenizer, device=torch_device)

# Set EOS token as pad token
tokenizer.pad_token = tokenizer.eos_token # <|endoftext|>
tokenizer.pad_token_id = tokenizer.eos_token_id # 50256

# Dataset
dataset = datasets.load_dataset("alistvt/coqa-stories")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

# # Preprocess data
# train_dataset = train_dataset.map(lambda examples: utils.preprocess_data(examples, tokenizer, max_seq_length), batched=True, batch_size=1, num_proc=8, remove_columns=train_dataset.column_names)
# eval_dataset = eval_dataset.map(lambda examples: utils.preprocess_data(examples, tokenizer, max_seq_length), batched=True, batch_size=1, num_proc=8, remove_columns=eval_dataset.column_names)
# eval_dataset = eval_dataset.select(range(eval_dataset_size)) # Small subset for quicker evaluation

# # Create data loaders
# train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=lambda batch: utils.collate_fn(batch, tokenizer.pad_token_id))
# eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=True, collate_fn=lambda batch: utils.collate_fn(batch, tokenizer.pad_token_id))

# Optimizer and scheduler
policy_optimizer = torch.optim.Adam(actor_critic_agent.policy_network.parameters(), lr=learning_rate)
value_optimizer = torch.optim.Adam(actor_critic_agent.value_network.parameters(), lr=learning_rate)
num_training_steps = num_episodes
policy_scheduler = transformers.get_linear_schedule_with_warmup(policy_optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
value_scheduler = transformers.get_linear_schedule_with_warmup(value_optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

# Train loop
for episode in tqdm(range(num_episodes)):
    state = "Hello, how are you?"
    encoded_state = actor_critic_agent.encode_sequence(state).unsqueeze(0).to(torch_device)
    current_state = encoded_state
    rewards, log_probs, states, actions = [], [], [], []

    for _ in tqdm(range(max_seq_length)):
        # Take action
        action, log_prob = actor_critic_agent.get_action_and_log_prob_rl(current_state)
        action = action.unsqueeze(0).to(torch_device)

        # Get reward
        states.append(current_state)
        current_state = torch.cat((current_state, action), dim=-1)
        decoded_state = actor_critic_agent.decode_sequence(current_state)
        decoded_state = ' '.join(decoded_state)
        reward = env.compute_rl_reward([decoded_state])

        # Append reward and log probability
        rewards.append(reward)
        log_probs.append(log_prob)
        actions.append(action)

    # Calculate losses
    loss, value_loss = actor_critic_agent.compute_loss_ppo_rl(states=states, rewards=rewards, old_log_probs=log_probs, actions=actions)
    print(f"Loss: {loss.item()}, Value Loss: {value_loss.item()}")

    # Policy Backward pass
    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    policy_scheduler.step()

    # Value Backward pass
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    value_scheduler.step()

    # Log the losses, their percentages, the learning rate, and the epoch loss to wandb
    common_utils.log_wandb({"Loss": loss.item(), "Value Loss": value_loss.item(), "Learning Rate": policy_optimizer.param_groups[0]['lr']})

    if episode % save_steps == 0 and episode != 0:
        # Save model checkpoint
        torch.save(actor_critic_agent.state_dict(), f'outputs/checkpoint_{episode}.pt')

    print(f'Episode {episode}: Loss {loss.item()}, Value Loss: {value_loss.item()}')

# Save model at the end of training
torch.save(actor_critic_agent.state_dict(), f'outputs/checkpoint_{episode}_final.pt')
print("Training completed.")