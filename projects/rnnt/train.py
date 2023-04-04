# Imports
import torch
import string
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
import unidecode
import IPython

# Hyperparameters
NULL_INDEX = 0

encoder_dim = 1024
predictor_dim = 1024
joiner_dim = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

# Encoder network
# The encoder is any network that can take as input a variable-length sequence: so, RNNs, CNNs, and self-attention/Transformer encoders will all work.


class Encoder(torch.nn.Module):
  def __init__(self, num_inputs):
    super(Encoder, self).__init__()
    self.embed = torch.nn.Embedding(num_inputs, encoder_dim)
    self.rnn = torch.nn.GRU(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
    self.linear = torch.nn.Linear(encoder_dim*2, joiner_dim)

  def forward(self, x):
    out = x
    out = self.embed(out)
    out = self.rnn(out)[0]
    out = self.linear(out)
    return out
     

# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
#         super(Encoder, self).__init__()
#         self.pos_encoder = PositionalEncoding(input_size)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(input_size))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src, mask):
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_key_padding_mask=mask)
#         output = self.dropout(output)
#         return output

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
    

# Predictor network
# The predictor is any causal network (= can't look at the future): in other words, unidirectional RNNs, causal convolutions, or masked self-attention.


class Predictor(torch.nn.Module):
  def __init__(self, num_outputs):
    super(Predictor, self).__init__()
    self.embed = torch.nn.Embedding(num_outputs, predictor_dim)
    self.rnn = torch.nn.GRUCell(input_size=predictor_dim, hidden_size=predictor_dim)
    self.linear = torch.nn.Linear(predictor_dim, joiner_dim)
    
    self.initial_state = torch.nn.Parameter(torch.randn(predictor_dim))
    self.start_symbol = NULL_INDEX # In the original paper, a vector of 0s is used; just using the null index instead is easier when using an Embedding layer.

  def forward_one_step(self, input, previous_state):
    embedding = self.embed(input)
    state = self.rnn.forward(embedding, previous_state)
    out = self.linear(state)
    return out, state

  def forward(self, y):
    batch_size = y.shape[0]
    U = y.shape[1]
    outs = []
    state = torch.stack([self.initial_state] * batch_size).to(y.device)
    for u in range(U+1): # need U+1 to get null output for final timestep 
      if u == 0:
        decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
      else:
        decoder_input = y[:,u-1]
      out, state = self.forward_one_step(decoder_input, state)
      outs.append(out)
    out = torch.stack(outs, dim=1)
    return out
     

# class MaskedSelfAttention(nn.Module):
#     def __init__(self, input_size, hidden_size, num_heads, dropout):
#         super(MaskedSelfAttention, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.attention = nn.MultiheadAttention(input_size, num_heads, dropout)
#         self.layer_norm = nn.LayerNorm(input_size)

#     def forward(self, x, mask):
#         # x: (seq_len, batch_size, input_size)
#         x = x.transpose(0, 1) # (batch_size, seq_len, input_size)
#         self.attention_output, _ = self.attention(x, x, x, attn_mask=mask)
#         # attention_output: (batch_size, seq_len, input_size)
#         x = x + self.attention_output
#         x = self.layer_norm(x)
#         x = x.transpose(0, 1) # (seq_len, batch_size, input_size)
#         return x

# class Predictor(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
#         super(Predictor, self).__init__()
#         self.attention_layers = nn.ModuleList([
#             MaskedSelfAttention(input_size, hidden_size, num_heads, dropout) for _ in range(num_layers)
#         ])
#         self.feedforward_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(input_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_size, input_size),
#                 nn.Dropout(dropout),
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, x, mask):
#         for attention_layer, feedforward_layer in zip(self.attention_layers, self.feedforward_layers):
#             x = attention_layer(x, mask)
#             x = feedforward_layer(x)
#         return x
    

# Joiner network
# The joiner is a feedforward network/MLP which sums the predictor and encoder output and runs it through a single linear layer

# class Joiner(nn.Module):
#     def __init__(self, input_size, out_dim):
#         super(Joiner, self).__init__()
#         self.linear = nn.Linear(input_size, out_dim)

#     def forward(self, encoder_output, predictor_output):
#         x = encoder_output + predictor_output
#         x = self.linear(x)
#         # softmax
#         x = F.log_softmax(x, dim=-1)
#         return x
    

class Joiner(torch.nn.Module):
  def __init__(self, num_outputs):
    super(Joiner, self).__init__()
    self.linear = torch.nn.Linear(joiner_dim, num_outputs)

  def forward(self, encoder_out, predictor_out):
    out = encoder_out + predictor_out
    out = torch.nn.functional.relu(out)
    out = self.linear(out)
    return out


# Loss function
class Transducer(torch.nn.Module):
  def __init__(self, num_inputs, num_outputs):
    super(Transducer, self).__init__()
    self.encoder = Encoder(num_inputs)
    self.predictor = Predictor(num_outputs)
    self.joiner = Joiner(num_outputs)

    if torch.cuda.is_available(): self.device = "cuda:0"
    else: self.device = "cpu"
    self.to(self.device)

  def compute_forward_prob(self, joiner_out, T, U, y):
    """
    joiner_out: tensor of shape (B, T_max, U_max+1, #labels)
    T: list of input lengths
    U: list of output lengths 
    y: label tensor (B, U_max+1)
    """
    B = joiner_out.shape[0]
    T_max = joiner_out.shape[1]
    U_max = joiner_out.shape[2] - 1
    log_alpha = torch.zeros(B, T_max, U_max+1).to(model.device)
    for t in range(T_max):
      for u in range(U_max+1):
          if u == 0:
            if t == 0:
              log_alpha[:, t, u] = 0.

            else: #t > 0
              log_alpha[:, t, u] = log_alpha[:, t-1, u] + joiner_out[:, t-1, 0, NULL_INDEX] 
                  
          else: #u > 0
            if t == 0:
              log_alpha[:, t, u] = log_alpha[:, t,u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
            
            else: #t > 0
              log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                  log_alpha[:, t-1, u] + joiner_out[:, t-1, u, NULL_INDEX],
                  log_alpha[:, t, u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
              ]), dim=0)
    
    log_probs = []
    for b in range(B):
      log_prob = log_alpha[b, T[b]-1, U[b]] + joiner_out[b, T[b]-1, U[b], NULL_INDEX]
      log_probs.append(log_prob)
    log_probs = torch.stack(log_probs) 
    return log_prob

  def compute_loss(self, x, y, T, U):
    encoder_out = self.encoder.forward(x)
    predictor_out = self.predictor.forward(y)
    joiner_out = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)
    loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()
    return loss
  
def compute_single_alignment_prob(self, encoder_out, predictor_out, T, U, z, y):
    """
    Computes the probability of one alignment, z.

    Target: Hypothetical target
    Joiner_out: Softmax of the joiner network
    """
    t = 0; u = 0
    t_u_indices = []
    y_expanded = []
    for step in z:
      t_u_indices.append((t,u))
      if step == 0: # right (null)
        y_expanded.append(NULL_INDEX)
        t += 1
      if step == 1: # down (label)
        y_expanded.append(y[u])
        u += 1
    t_u_indices.append((T-1,U))
    y_expanded.append(NULL_INDEX)

    t_indices = [t for (t,u) in t_u_indices]
    u_indices = [u for (t,u) in t_u_indices]
    encoder_out_expanded = encoder_out[t_indices] # encoder output 
    predictor_out_expanded = predictor_out[u_indices]
    joiner_out = self.joiner.forward(encoder_out_expanded, predictor_out_expanded).log_softmax(1)

    # target is the observed outputs and joiner_output is the probability distributions for each timestep.
    logprob = -torch.nn.functional.nll_loss(input=joiner_out, target=torch.tensor(y_expanded).long().to(self.device), reduction="sum")
    return logprob

Transducer.compute_single_alignment_prob = compute_single_alignment_prob


# Generate example inputs/outputs
num_outputs = len(string.ascii_uppercase) + 1 # [null, A, B, ... Z]
model = Transducer(1, num_outputs).to(device)
y_letters = "CAT"
y = torch.tensor([string.ascii_uppercase.index(l) + 1 for l in y_letters]).unsqueeze(0).to(model.device)
T = torch.tensor([4]) # Input sequence length
U = torch.tensor([len(y_letters)]) # Output sequence length
B = 1 # Batch

encoder_out = torch.randn(B, T, joiner_dim).to(model.device)
predictor_out = torch.randn(B, U+1, joiner_dim).to(model.device) # TODO: I don't understand the U+1
joiner_out = model.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3) # TODO I don't understand why we need all the unsqueezing 

#######################################################
# Compute loss by enumerating all possible alignments #
#######################################################
all_permutations = list(itertools.permutations([0]*(T-1) + [1]*U)) # TODO: I don't understand the T-1
all_distinct_permutations = list(Counter(all_permutations).keys())
alignment_probs = []

# the Transducer defines p(y|x) = the sum of the probabilities of all possible alignments between x and y.
for z in all_distinct_permutations:
  alignment_prob = model.compute_single_alignment_prob(encoder_out[0], predictor_out[0], T.item(), U.item(), z, y[0])
  alignment_probs.append(alignment_prob)

loss_enumerate = -torch.tensor(alignment_probs).logsumexp(0)

#######################################################
# Compute loss using the forward algorithm            #
#######################################################

# TODO up to here - understanding what this is doing.
loss_forward = -model.compute_forward_prob(joiner_out, T, U, y)

print("Loss computed by enumerating all possible alignments: ", loss_enumerate)
print("Loss computed using the forward algorithm: ", loss_forward)
     