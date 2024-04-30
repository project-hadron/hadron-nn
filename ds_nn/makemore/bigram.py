import torch
import torch.nn.functional as F
import torch.nn as nn

# hyperparameters
batch_size = 32
block_size = 8
max_iter = 1000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 2000
# number of embedding dimensions
n_embed = 32

torch.manual_seed(1337)

# read all the words
with open('./source/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# build the vocabulary of characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# tokenizer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# split the data
n = int(0.9*len(data))
train_data = data[n:]
val_data = data[:n]

# get data chunks
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BiogramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # embedding wrapper but adding n_embed as an intermediate reference rather that direct to vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # to get to our logits we create a Linear layer to return a learning model head (lm_head)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # not only the Embedding of the identity (idx), but also each position of t in block_size
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # with n_embed we no longer reference logits directly but must go through a Linear layer
        token_emb = self.token_embedding_table(idx)  # (Batch, Time, n_embed)
        poss_emb = self.position_embedding_table(torch.arange(T, device=device))  # (Time, n_embed)
        # x now becomes the token embedding with the positional embedding in the batch
        x = token_emb + poss_emb  # (Batch, Time, n_embed)
        # the linear layer then returns the logit index and position in the batch
        logits = self.lm_head(x)  # (Batch, Time, Vocab_size)
        if targets is None:
            loss = None
        else:
            # torch cross_entropt expects B C T if three dimentional so make it 2 dimentions
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # targets to match logits
            targets = targets.view(B*T)
            # cross_entropy of output (logits) against target labels
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # use softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # returns (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # returns (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # returns (B, T+1)
        return idx

model = BiogramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# train loop
for step in range(eval_iters):

    # print out the loss
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

if __name__ == "__main__":
    print('run done')