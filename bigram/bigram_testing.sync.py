# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
block_size = 8
batch_size = 4
max_iters = 100000
learning_rate = 3e-4
eval_every = 5000

# %%
with open("shakespeare.txt") as f:
    text = f.read()
print(text[:500])

# %%
chars = sorted(set(text))
vocab_size = len(chars)

# %%
print(f"Vocab size: {vocab_size}")
print(f"Text length: {len(text)}")

# %%
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[ch] for ch in s]
decode = lambda x: "".join([int_to_string[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long, device=device)


# %%
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


# %%
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# %%
x, y = get_batch("train")

# %%


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(
                B * T, C
            )  # reshape to what torch.cross_entropy expects
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index


# %%
model = BigramLanguageModel(vocab_size).to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=100)[0].tolist())
print(generated_chars)

# %% [markdown]
#
# ### Some common optimizers
# 1. **Mean Squared Error (MSE)**: MSE is a common loss function used in regression problems, where the goal is to predict a continuous output. It measures the average squared difference between the predicted and actual values, and is often used to train neural networks for regression tasks.
# 2. **Gradient Descent (GD):**  is an optimization algorithm used to minimize the loss function of a machine learning model. The loss function measures how well the model is able to predict the target variable based on the input features. The idea of GD is to iteratively adjust the model parameters in the direction of the steepest descent of the loss function
# 3. **Momentum**: Momentum is an extension of SGD that adds a \"momentum\" term to the parameter updates. This term helps smooth out the updates and allows the optimizer to continue moving in the right direction, even if the gradient changes direction or varies in magnitude. Momentum is particularly useful for training deep neural networks.
# 4. **RMSprop**: RMSprop is an optimization algorithm that uses a moving average of the squared gradient to adapt the learning rate of each parameter. This helps to avoid oscillations in the parameter updates and can improve convergence in some cases.
# 5. **Adam**: Adam is a popular optimization algorithm that combines the ideas of momentum and RMSprop. It uses a moving average of both the gradient and its squared value to adapt the learning rate of each parameter. Adam is often used as a default optimizer for deep learning models.
# 6. **AdamW**: AdamW is a modification of the Adam optimizer that adds weight decay to the parameter updates. This helps to regularize the model and can improve generalization performance. We will be using the AdamW optimizer as it best suits the properties of the model we will train in this video.

# find more optimizers and details at torch.optim

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # sample a batch
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_every == 0:
        print(f"Iter {iter}:")
        print(loss.item())
print(loss.item())

# %%

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=100)[0].tolist())
print(generated_chars)
