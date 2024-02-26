### import libraries
import numpy as np
import pandas as pd
import torch
import json
import regex as re
import clean_txt as clean
import os

### Build word ditionary
def dictionary(filepath):
    with open(filepath+"training_label.json") as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            word_sentence = clean.clean_txt(s, w2v=True)
            word_sentence = [word for word in word_sentence.split()]
            for word in word_sentence:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    word_dict = {}
    for word in word_count:
        if word_count[word] > 4:
            word_dict[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<BOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, word_dict

# return a list of indices given text
def text_to_indices(text, w2i, i2w, word_dict):
    text = clean.clean_txt(text, w2v=True)
    text = text.split()
    sentence = []
    for word in text:
        if word not in word_dict:
            sentence.append(i2w[3])
        else:
            sentence.append(word)
    text = [w2i[word] for word in sentence]
    return text

# pad and add bos eos tags
def add_tags(text, pad_len):
    text.insert(0,1)
    text.append(2)
    while len(text) < pad_len+2:
        text.append(0)
    return text

### One hot encoding
def one_hot_encoding(indices, num_classes):
    one_hot = torch.zeros(len(indices), num_classes)
    one_hot.scatter_(1, torch.tensor(indices).unsqueeze(1), 1)
    return one_hot

### Feature load
def feature(filepath):
    feats = {}
    training_feats = filepath + 'training_data/feat'
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        feats[file.split('.npy')[0]] = value
    return feats

### training list of [id, caption, feature]
def get_training_list(filepath, w2i, i2w, word_dict):
    with open(filepath + "training_label.json") as f:
        cap = json.load(f)
    #print(len(cap))
    training_list = []
    pad_len = 0
    captions = []
    for i in range(len(cap)-1):
        for caption in range(len(cap[i]['caption'])-1):
            #one_hot_encoded = one_hot_encoding(indices = text_to_indices(cap[i]['caption'][caption], w2i), num_classes = len(word_dict))
            #training_list.append([cap[i]['id'], one_hot_encoded])
            text = text_to_indices(cap[i]['caption'][caption], w2i, i2w, word_dict)
            captions.append([cap[i]['id'], text])
            if len(text) > pad_len:
                pad_len = len(text)
    for caption in captions:
        #print(caption[1],[i2w[word] for word in add_tags(caption[1], pad_len=pad_len)])
        one_hot_encoded = one_hot_encoding(indices = add_tags(caption[1], pad_len=pad_len), num_classes = len(word_dict)+4)
        training_list.append([caption[0], one_hot_encoded])
    features = feature(filepath="../MLDS_hw2_1_data/")
    for label in training_list:
        label.append(features[label[0]])
    return training_list   

### Main
filepath = "../MLDS_hw2_1_data/"
i2w, w2i, word_dict = dictionary(filepath=filepath)
#features = feature(filepath="../MLDS_hw2_1_data/")
train_list = get_training_list(filepath=filepath, w2i=w2i, i2w=i2w, word_dict=word_dict)
for i in train_list[0][1:]:
    print(i.shape)
