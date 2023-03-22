# Download input data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
import torch.nn as nn
from torch.nn import functional as F
import IPython
from tqdm import tqdm

torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8  # We still train with all smaller context sizes but this is the maximum. 0 --> 1, 0,1 --> 2, 0,1,2 -->3 ... 0...n-1 --> n
max_iters = 2000
learning_rate = 1e-2
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print("device", device)
eval_iters = 200
eval_interval = 300


# Read the input file
with open("./input.txt", "r", encoding="utf-8") as f:
    text = f.read()  # should be simple plain text file
print("corpus length:", len(text))

# Create character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("total chars:", vocab_size)
print("".join(chars))

# Tokenize the text
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: "".join([itos[ch] for ch in x])

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
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([train_data[i : i + block_size] for i in ix])
    y = torch.stack([train_data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# Eval
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    """Bigram LM because we only look at one character of context to make predictions for the next char."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(
            idx
        )  # (Batch,Time,Channel), acting as a lookup table for the probabilities of the next word
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(
                logits, dim=-1
            )  # final dimension as that's the dimension with the activation values.
            idx_next = torch.multinomial(probs, num_samples=1, replacement=False)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)
print("model is on", next(m.parameters()).device)
# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters)):
    # get batch
    xb, yb = get_batch("train")

    # eval
    if iter % eval_interval == 0:
        loss = estimate_loss()
        print("iter ", iter, loss)
    # forward pass
    logits, loss = m(xb, yb)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


# Sample start input
input = torch.zeros((1, 1), dtype=torch.long).to(device)
out = decode(
    m.generate(input, max_new_tokens=500)[0].tolist()
)  # [0] is needed as we have a Batch dimension
print(out)
