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
import transformers

# Set a seed for the random number generator to ensure reproducibility
random.seed(0)

import sys
sys.path.append("../..")
import common.utils as common_utils

# Hyperparameters
experiment_name = "base"
num_episodes = 10
max_seq_length = 100
learning_rate = 4e-3
device = "cuda"
eval_steps = 100
save_steps = 500
do_eval = True
train_batch_size = 24
eval_batch_size = 24
eval_dataset_size = 96
max_episode_length = 5
warmup_steps = 100
# TODO: Add episolon gamma, beta from agent

# Create outputs folder
common_utils.create_folder("outputs")

# Start wandb logging
# common_utils.start_wandb_logging(project_name="llm_rl_finetuning", name=experiment_name)

# Initialize environment and agent
torch_device = common_utils.get_device(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(tokenizer=tokenizer, max_seq_length=max_seq_length, device=torch_device)
simple_agent = agent.ActorCriticAgent(model=model, tokenizer=tokenizer)

# Set EOS token as pad token
tokenizer.pad_token = tokenizer.eos_token # <|endoftext|>
tokenizer.pad_token_id = tokenizer.eos_token_id # 50256

# Move model components to device
model.to(torch_device)

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
policy_optimizer = torch.optim.Adam(simple_agent.policy_network.parameters(), lr=learning_rate)
value_optimizer = torch.optim.Adam(simple_agent.value_network.parameters(), lr=learning_rate)
# num_training_steps = num_episodes * len(train_dataloader)
# scheduler = transformers.get_linear_schedule_with_warmup(policy_optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

# Train loop
episode_rewards, episode_log_probs, episode_states, episode_actions = [], [], [], []

for episode in range(num_episodes):
    state = "Hello, how are you?"
    encoded_state = simple_agent.encode_sequence(state).unsqueeze(0).to(torch_device)
    current_state = encoded_state
    done = False
    rewards, log_probs, states, actions = [], [], [], []

    while not done:
        # Take action
        action, log_prob = simple_agent.get_action_and_log_prob_rl(current_state)
        action = action.unsqueeze(0).to(torch_device)

        # Get reward
        states.append(current_state)
        current_state = torch.cat((current_state, action), dim=-1)
        decoded_state = simple_agent.decode_sequence(current_state)
        decoded_state = ' '.join(decoded_state)
        reward = env.compute_rl_reward([decoded_state])

        # Append reward and log probability
        rewards.append(reward)
        log_probs.append(log_prob)
        actions.append(action)

        # Episode termination condition
        if len(rewards) >= max_episode_length:
            done = True

    episode_rewards.append(rewards)
    episode_log_probs.append(log_probs)
    episode_states.append(states)
    episode_actions.append(actions)

    # Policy update
    loss, value_loss = simple_agent.compute_loss_ppo_rl(states=episode_states[-1], rewards=episode_rewards[-1], old_log_probs=episode_log_probs[-1], actions=episode_actions[-1])
    print(f"Loss: {loss.item()}, Value Loss: {value_loss.item()}")

    # Backward pass
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    loss.backward()
    value_optimizer.step()
    policy_optimizer.step()
    # scheduler.step()


    #     classifier_loss_percentage = (mean_classifier_loss / loss) * 100
    #     learning_rate = scheduler.get_last_lr()[0]

    #     # Log the losses, their percentages, the learning rate, and the epoch loss to wandb
    #     common_utils.log_wandb({"classifier_loss": mean_classifier_loss, "cross_entropy_loss": ce_loss, "classifier_loss_percentage": classifier_loss_percentage, "learning_rate": learning_rate, "epoch": epoch, "total_loss": loss})

    #     if step % save_steps == 0 and step != 0:
    #         # Save model checkpoint
    #         torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_{step}.pt')

    # print(f'Epoch {epoch}: Loss {loss.item()}')
    # # Save model at the end of every epoch
    # torch.save(model.state_dict(), f'outputs/checkpoint_{epoch}_final.pt')




# import torch
# import random
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# from torch.distributions import Categorical

# # Initialize GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# config = GPT2Config.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# # Initialize optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Function to get action and log probability
# def get_action_and_log_prob(state, model, epsilon=0.1):
#     inputs = tokenizer(state, return_tensors="pt")
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.softmax(logits, dim=-1)
    
#     # Epsilon-greedy exploration strategy
#     if random.random() < epsilon:
#         action = torch.randint(0, logits.shape[-1], (1,))
#     else:
#         dist = Categorical(probs)
#         action = dist.sample()
    
#     # Log probability for training
#     dist = Categorical(probs)
#     log_prob = dist.log_prob(action)
    
#     return action.item(), log_prob

# # PPO update function
# def update_model(states, rewards, log_probs, gamma=0.99, epsilon=0.2):
#     R = 0
#     discounted_rewards = []
    
#     # Compute discounted rewards
#     for r in reversed(rewards):
#         R = r + gamma * R
#         discounted_rewards.insert(0, R)
    
#     # Update policy by PPO
#     log_probs = torch.stack(log_probs)
#     discounted_rewards = torch.Tensor(discounted_rewards)
#     advantages = discounted_rewards - log_probs.exp()
    
#     ratio = (log_probs - log_probs.detach()).exp()
#     surr1 = ratio * advantages
#     surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    
#     loss = -torch.min(surr1, surr2).mean()
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # RL Training loop
# episode_rewards = []
# episode_log_probs = []
# episode_states = []

# for episode in range(10):  # Running for 10 num_episodes as an example
#     state = "Hello, how are you?"  # Initialize state (In a more complex scenario, you may have dynamic states)
#     done = False
#     rewards = []
#     log_probs = []
    
#     while not done:
#         action, log_prob = get_action_and_log_prob(state, model)
        
#         # Placeholder for reward assignment logic
#         # Here, you might call your environment to get a reward
#         # For demonstration, we are using random rewards
#         reward = random.choice([0, 1])
        
#         rewards.append(reward)
#         log_probs.append(log_prob)
        
#         # Episode termination condition (For demonstration, we're using a fixed length)
#         if len(rewards) >= 5:
#             done = True
    
#     episode_rewards.append(rewards)
#     episode_log_probs.append(log_probs)
#     episode_states.append(state)
    
#     # Policy update
#     update_model(episode_states, episode_rewards[-1], episode_log_probs[-1])
