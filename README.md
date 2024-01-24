---
title: LLM From Scratch
emoji: 🧠
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# LLM From Scratch
_Ibrahim Khalid_

The hosted project is available on [HuggingFace](https://huggingface.co/spaces/ibrahimmkhalid/llm-from-scratch)  

The purpose of this project is to build a simple large language model from scratch.


This repo is following the guide from https://www.youtube.com/watch?v=UU1WVnMk4E8

In this repo:
- ./shakespeare.txt - This is a sample text used for training a smaller scale model
- ./bigram_testing.sync.ipynb - This notebook is where I test a basic BiGram model
- ./gpt_shakespeare.sync.ipynb - Notebook implementing simple GPT model using entire works of shakespeare
- ./gpt_openwebtext.sync.ipynb - Notebook implementing GPT model based on the [OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)


### Prepare environment
```
pip install -r ./requirements-base.txt  
pip install -r ./requirements-pytorch.txt
```
---
[On GitHub](https://github.com/ibrahimmkhalid/llm-from-scratch)  

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
