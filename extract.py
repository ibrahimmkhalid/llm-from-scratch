import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

tarxz_path = "./openwebtext.tar.xz"
folder_path = "./openwebtext"
output_file_train = "./openwebtext/train_split.txt"
output_file_val = "./openwebtext/val_split.txt"
vocab_file = "./vocab.txt"

if not os.path.exists(tarxz_path):
    print("Please download the openwebtext.tar.xz file from:")
    print("https://skylion007.github.io/OpenWebTextCorpus/")
    exit()

# Extract the tar.xz file
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
    os.system(f"tar -xvf {tarxz_path}")

files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate the split indices
split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

# Process the files for training and validation separately
vocab = set()

# Process the training files
if not os.path.exists(output_file_train):
    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_train, total=len(files_train)):
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

# Process the validation files
if not os.path.exists(output_file_val):
    with open(output_file_val, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files_val, total=len(files_val)):
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)

# Write the vocabulary to vocab.txt
if not os.path.exists(vocab_file):
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in vocab:
            vfile.write(char + '\n')
