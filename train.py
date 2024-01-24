import torch
import mmap
import random
import os
from GPTLanguageModelClass import *

block_size = hyperparams.block_size
batch_size = hyperparams.batch_size
max_iters = hyperparams.max_iters
learning_rate = hyperparams.learning_rate
eval_every = hyperparams.eval_every
n_embd = hyperparams.n_embd
n_head = hyperparams.n_head
n_layer = hyperparams.n_layer
dropout = hyperparams.dropout
device = hyperparams.device

print(device)

if not os.path.exists("./vocab.txt") or not os.path.exists("./openwebtext/train_split.txt") or not os.path.exists("./openwebtext/val_split.txt"):
    raise Exception("Please run extract.py first")
chars = ""
with open("./vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
        
vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[ch] for ch in s]
decode = lambda x: ''.join([int_to_string[i] for i in x])
# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "./openwebtext/train_split.txt" if split == 'train' else "./openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_every)
        for k in range(eval_every):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(vocab_size).to(device)

model_pickle_path = './model.pt'
if os.path.exists(model_pickle_path):
    print('loading model parameters...')
    with open(model_pickle_path, 'rb') as f:
        model = torch.load(f, map_location=device)
    print('loaded successfully!')
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_every == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open(model_pickle_path, 'wb') as f:
    torch.save(model, f)
print('model saved')
