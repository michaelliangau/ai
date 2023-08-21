from transformers import GPT2LMHeadModel, GPT2Tokenizer
import agent
import environment
import IPython

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
env = environment.Environment(max_length=100)
ag = agent.PPOAgent(model, tokenizer)
ag.train(env)