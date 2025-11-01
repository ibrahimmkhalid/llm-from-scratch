import streamlit as st
import torch
import os
from GPTLanguageModelClass import hyperparams

st.set_page_config(page_title="LLM from Scratch Demo")
st.title("LLM from Scratch Demo")

block_size = hyperparams.block_size
device = hyperparams.device

if not os.path.exists("./vocab.txt"):
    st.error("Please run extract.py first")
    st.stop()

with open("./vocab.txt", "r", encoding="utf-8") as f:
    chars = sorted(list(set(f.read())))

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [string_to_int[ch] for ch in s]


def decode(x):
    return "".join([int_to_string[i] for i in x])


@st.cache_resource
def load_model():
    model_pickle_path = "./model.pt"
    with open(model_pickle_path, "rb") as f:
        model = torch.load(f, map_location=device, weights_only=False)
    return model


model = load_model()

if "result" not in st.session_state:
    st.session_state.result = None

if "prompt" not in st.session_state:
    st.session_state.prompt = ""


def clear_results():
    st.session_state.result = None
    st.session_state.prompt = ""


st.subheader("About")

st.markdown(
    'This is a demo of a language model built from scratch using PyTorch. It generates text continuations based on a *character*-level GPT architecture trained on the [OpenWebText dataset](https://github.com/jcpeterson/openwebtext). What this means is that this model will "predict" the next character based on all previous characters. This model was built from scratch using PyTorch, following the [paper](https://arxiv.org/abs/1706.03762) "Attention is all you need". The goal of this project was to gain a deep familiarity with the underlying structure of an LLM. The model was trained on commodity hardware and utilized a comparatively small dataset size and model size.'
)

st.subheader("Model")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Device:** {device}")
    st.write(f"**Vocab size:** {len(chars)}")
    st.write(f"**Block size:** {block_size}")
    st.write(f"**Batch size:** {hyperparams.batch_size}")
with col2:
    st.write(f"**Max iters:** {hyperparams.max_iters}")
    st.write(f"**Learning rate:** {hyperparams.learning_rate}")
    st.write(f"**Eval every:** {hyperparams.eval_every}")
    st.write(f"**n_embd:** {hyperparams.n_embd}")
with col3:
    st.write(f"**n_head:** {hyperparams.n_head}")
    st.write(f"**n_layer:** {hyperparams.n_layer}")
    st.write(f"**Dropout:** {hyperparams.dropout}")


st.subheader("Demo")
st.write(
    "Enter some text (up to 127 characters) and click 'Generate' to see "
    "the model's continuation"
)

prompt = st.text_area(
    "Enter text to autocomplete:",
    height=50,
    max_chars=block_size - 1,
    key="prompt",
    placeholder="Type here...",
)

generate_clicked = st.button("Generate")
clear_clicked = st.button("Clear Results", on_click=clear_results)

if generate_clicked or len(prompt) != 0:
    if prompt.strip():
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        max_new_tokens = block_size - len(prompt)
        generated = model.generate(context.unsqueeze(0), max_new_tokens=max_new_tokens)[
            0
        ]
        full_text = decode(generated.tolist())
        st.session_state.result = {
            "input": prompt,
            "continuation": full_text[len(prompt) :],
            "full": full_text,
        }
    else:
        st.warning("Please enter some text to autocomplete.")
        st.session_state.result = None

if st.session_state.result:
    st.subheader("Result")
    st.write("**Your input:**")
    st.write(st.session_state.result["input"])
    st.write("**Generated continuation:**")
    st.write(st.session_state.result["continuation"])
    st.write("**Full text:**")
    st.write(st.session_state.result["full"])

st.markdown("---")
st.markdown(
    "Connect with me"
    ": [GitHub](https://github.com/ibrahimmkhalid/llm-from-scratch) "
    "| [LinkedIn](https://linkedin.com/in/ibrahimmkhalid) "
    "| [Website](https://ibrahimkhalid.me) "
    "| [ibrahimmkhalid@gmail.com](mailto:ibrahimmkhalid@gmail.com)"
)
