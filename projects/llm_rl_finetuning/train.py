from transformers import GPT2LMHeadModel, GPT2Tokenizer
import agent
import environment
import IPython

# Hyperparameters
epochs = 10
max_seq_length = 100

# Train
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(max_seq_length=max_seq_length)
ag = agent.PPOAgent(model, tokenizer)
ag.train(env, epochs=epochs)