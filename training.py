
### import libraries
import numpy as np
import pandas as pd
import torch
import json
import regex as re

### Build word ditionary
def dictionary(filepath):
    with open(filepath+"training_label.json") as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            word_sentence = re.sub('[.!,;?]]', ' ', s).split()
            for word in word_sentence:
                word = word.replace('.', '').lower() if '.' in word else word.lower()
                word = word.replace(',', '').lower() if ',' in word else word.lower()
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    word_dict = {}
    for word in word_count:
        if word_count[word] > 4:
            word_dict[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, word_dict

### One hot encoding
def one_hot_encoding(indices, num_classes):
    one_hot = torch.zeros(len(indices), num_classes)
    one_hot.scatter_(1, torch.tensor(indices).unsqueeze(1), 1)
    return one_hot

### Main
i2w, w2i, word_dict = dictionary(filepath= "../MLDS_hw2_1_data/")


# Example text
text = "A man is in the box"#"hello world python"

# Convert text to indices
indices = [w2i[word] for word in text.split()]

# Get the number of classes (vocabulary size)
num_classes = len(word_dict)

# Create one-hot encoding tensor
one_hot_encoded = one_hot_encoding(indices, num_classes)

print(one_hot_encoded)



