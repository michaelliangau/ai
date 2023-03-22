
# Download input data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
import torch.nn as nn
from torch.nn import functional as F
import IPython
from tqdm import tqdm
torch.manual_seed(1337)

# hyperparameters
batch_size = 64
block_size = 256 # We still train with all smaller context sizes but this is the maximum. 0 --> 1, 0,1 --> 2, 0,1,2 -->3 ... 0...n-1 --> n 
max_iters = 5000
learning_rate = 3e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('device', device)
eval_iters = 200
eval_interval = 1000
n_embd = 384
num_sa_heads = 6
dropout = 0.2
n_heads = 6
n_layers = 6


# Read the input file
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # should be simple plain text file
print('corpus length:', len(text))

# Create character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('total chars:', vocab_size)
print(''.join(chars))

# Tokenize the text
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[ch] for ch in x])

# print('encoded:', encode("hey there!"))
# print('decoded:', decode(encode("hey there!")))

# Encode the entire dataset
data = torch.tensor(encode(text)).to(device)

# Split dataset into train and validation dataset
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Dataloader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

# Eval
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    """Linear layer following self attention"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """Self attention block"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Buffer is used because tril is not a model parameter. It will not be stored in the state dict.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # scaled attention
        wei = wei.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple self attentions running in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Regular python list cannot return parameters thus cannot be trained.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C)
        x = self.proj(x)
        out = self.dropout(x)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """Bigram LM because we only look at one character of context to make predictions for the next char."""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (Batch,Time,Channel), acting as a lookup table for the probabilities of the next word
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (B, T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]# crop context so positioning embedding table doesn't error out
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # final dimension as that's the dimension with the activation values.
            idx_next = torch.multinomial(probs, num_samples=1, replacement=False)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



model = BigramLanguageModel()
m = model.to(device)
print('model is on', next(m.parameters()).device)
# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters)):
    # get batch
    xb, yb = get_batch('train')

    # eval
    if iter % eval_interval == 0:
        loss = estimate_loss()
        print('iter ', iter,  loss)
    # forward pass
    logits, loss = m(xb, yb)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('final loss', loss.item())


# Sample start input
input = torch.zeros((1,1), dtype=torch.long).to(device)
out = decode(m.generate(input, max_new_tokens=500)[0].tolist()) # [0] is needed as we have a Batch dimension

print(out)




