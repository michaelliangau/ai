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

# Set a seed for the random number generator to ensure reproducibility
random.seed(0)

import sys
sys.path.append("../..")
import common.utils as common_utils

# Hyperparameters
experiment_name = "dev"
num_episodes = 100
max_seq_length = 60
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
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# env = environment.Environment(tokenizer=tokenizer, device=torch_device)
actor_critic_agent = agent.ActorCriticAgent(tokenizer=tokenizer, device=torch_device)

IPython.embed()
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
]
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(device)
generated_ids = actor_critic_agent.policy_network.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)


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
    state = "Explain nuclear fusion like I'm five."
    encoded_state = actor_critic_agent.encode_sequence(sequence=state).unsqueeze(0).to(torch_device)
    current_state = encoded_state
    rewards, log_probs, states, actions = [], [], [], []

    for i in tqdm(range(max_seq_length)):
        # Take action
        action, log_prob = actor_critic_agent.get_action_and_log_prob_rl(state=current_state)
        action = action.unsqueeze(0).to(torch_device)

        # Get reward
        states.append(current_state)
        current_state = torch.cat((current_state, action), dim=-1)
        model_output = current_state[:, encoded_state.size(1):]
        decoded_model_text = actor_critic_agent.decode_sequence(sequence=model_output)
        decoded_model_text = ' '.join(decoded_model_text)
        reward = env.compute_rl_classifier_reward(model_output=[decoded_model_text])

        # Append reward and log probability
        rewards.append(reward)
        log_probs.append(log_prob)
        actions.append(action)

    # Compute RLHF reward across entire generated sequence
    actions_tensor = torch.cat(actions, dim=0).squeeze()
    actions_text = actor_critic_agent.decode_sequence(actions_tensor)
    actions_text = ''.join(actions_text)
    rlhf_rewards = env.compute_rlhf_reward(model_outputs=[actions_text], states=[state]) / 10

    # Add RLHF reward to rewards
    summed_rewards = [r + rlhf_rewards for r in rewards]

    # Calculate cumulative reward
    cumulative_reward = sum(summed_rewards)

    # Calculate RLHF reward percentage for metrics
    rlhf_reward_perc = (abs(rlhf_rewards.item() * len(actions)) / (abs(rlhf_rewards.item() * len(actions)) + abs(sum(rewards).item()))) * 100

    # Calculate losses
    policy_loss, value_loss = actor_critic_agent.compute_loss_ppo_rl(states=states, rewards=rewards, old_log_probs=log_probs, actions=actions)
    print(f"Policy Loss: {policy_loss.item()}")
    print(f"Value Loss: {value_loss.item()}")
    print(f"Cumulative Reward: {cumulative_reward.item()}")
    print(f"RLHF Reward %: {rlhf_reward_perc}")
    print(f"Every 10th classifier reward: {[rewards[i].item() for i in range(0, len(rewards), 10)]}")
    print(f"RLHF rewards: {rlhf_rewards.item()}")
    # Decode the generated sequence and print it out
    decoded_sequence = actor_critic_agent.decode_sequence(current_state)
    print(f"Decoded sequence: {decoded_sequence}")

    # Policy Backward pass
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    policy_scheduler.step()

    # Value Backward pass
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    value_scheduler.step()

    # Log the losses, their percentages, the learning rate, and the epoch loss to wandb
    common_utils.log_wandb({"Policy Loss": policy_loss.item(), "Value Loss": value_loss.item(), "Learning Rate": policy_optimizer.param_groups[0]['lr'], "Cumulative Reward": cumulative_reward, "RLHF Reward %": rlhf_reward_perc})

    if episode % save_steps == 0 and episode != 0:
        # Save model checkpoint
        torch.save(actor_critic_agent.state_dict(), f'outputs/checkpoint_{episode}.pt')

# Save model at the end of training
torch.save(actor_critic_agent.state_dict(), f'outputs/checkpoint_{episode}_final.pt')
print("Training completed.")