import streamlit as st
import torch
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

st.title('LLM from scratch Demo')

st.write(f"Using device: {device}")

if not os.path.exists("./vocab.txt"):
    raise Exception("Please run extract.py first")
chars = ""
with open("./vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[ch] for ch in s]
decode = lambda x: ''.join([int_to_string[i] for i in x])


model_pickle_path = './model.pt'

st.write('loading model parameters...')
with open(model_pickle_path, 'rb') as f:
    model = torch.load(f, map_location=device)
st.write('model loaded successfully!')

prompt = ''
prompt = st.text_area('Prompt:', value=prompt, height=100, max_chars=block_size - 1, key='prompt')
if len(prompt) != 0:
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    max_new_tokens = block_size - len(prompt)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=max_new_tokens)[0].tolist())
    st.write('Generated text:')
    st.write(generated_chars)
