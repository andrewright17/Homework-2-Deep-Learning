### import libraries
import numpy as np
import pandas as pd
import torch
import json
import regex as re
import clean_txt as clean

### Build word ditionary
def dictionary(filepath):
    with open(filepath+"training_label.json") as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            #word_sentence = re.sub('[.!,;?]]', ' ', s).split()
            word_sentence = clean.clean_txt(s, w2v=True)
            word_sentence = [word for word in word_sentence.split()]
            for word in word_sentence:
                #word = word.replace('.', '').lower() if '.' in word else word.lower()
                #word = word.replace(',', '').lower() if ',' in word else word.lower()
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    word_dict = {}
    for word in word_count:
        word_dict[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, word_dict

def text_to_indices(text, w2i):
    text = clean.clean_txt(text, w2v=True)
    return [w2i[word] for word in text.split()]

### One hot encoding
def one_hot_encoding(indices, num_classes):
    one_hot = torch.zeros(len(indices), num_classes)
    one_hot.scatter_(1, torch.tensor(indices).unsqueeze(1), 1)
    return one_hot

### Main
i2w, w2i, word_dict = dictionary(filepath= "../MLDS_hw2_1_data/")

# Example text
text = "A man is in the box"#"hello world python"

# Create one-hot encoding tensor
#one_hot_encoded = one_hot_encoding(indices = text_to_indices(text, w2i), num_classes = len(word_dict))

#print(one_hot_encoded)
#print(w2i)
### load feature ids
filepath= "../MLDS_hw2_1_data/"
with open(filepath + "training_data/id.txt") as f:
    data = f.read()

with open(filepath + "training_label.json") as f:
    cap = json.load(f)
#print(len(cap))
training_list = []
for i in range(len(cap)-1):
    for caption in range(len(cap[i]['caption'])-1):
        #print(f"\nid: {cap[i]['id']}, caption: {cap[i]['caption'][caption]}")
        one_hot_encoded = one_hot_encoding(indices = text_to_indices(cap[i]['caption'][caption], w2i), num_classes = len(word_dict))
        if len(one_hot_encoded) < 6:
            print(f"\nid: {cap[i]['id']}, caption: {cap[i]['caption'][caption]}")