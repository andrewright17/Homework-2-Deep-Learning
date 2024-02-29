### import libraries
import numpy as np
import pandas as pd
import random
from scipy.special import expit
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import json
import regex as re
import clean_txt as clean
import os

### Build word ditionary
def dictionary(filepath, label_file):
    with open(filepath+label_file) as f:
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
    text.insert(0,1)
    text.append(2)
    return text

# returns list of tuples (id, indexed caption)
def annotate(filepath, label_file, word_dict, w2i, i2w):
    label_json = filepath + label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = text_to_indices(s,  w2i, i2w, word_dict)
            annotated_caption.append((d['id'], s))
    return annotated_caption

# load the feature data 
def get_avi(filepath, files_dir):
    avi_data = {}
    training_feats = filepath + files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

# class to load training data
class training_data(Dataset):
    def __init__(self, filepath, label_file, files_dir, word_dict, w2i, i2w):
        self.filepath = filepath
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = get_avi(files_dir)
        self.w2i = w2i
        self.i2w = i2w
        self.data_pair = annotate(filepath, label_file, word_dict, w2i, i2w)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000. # add noise for regularization
        return torch.Tensor(data), torch.Tensor(sentence)

# batch data for training and padding
def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

# Attention Class
class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return nn.functional.softmax(attention, dim=1)

# Encoder Class
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
    
        self.linear = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.linear(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, hidden_state = self.lstm(input)

        return output, hidden_state

# Decoder Class
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(DecoderWithAttention, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, decoder_current_hidden_state = self.lstm(lstm_input, decoder_current_hidden_state)
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, decoder_current_hidden_state = self.lstm(lstm_input, decoder_current_hidden_state)
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85)) # inverse of the logit function

### Main
filepath = "../MLDS_hw2_1_data/"
files_dir = 'training_data/feat'
label_file = 'training_label.json'
i2w, w2i, word_dict = dictionary(filepath=filepath, label_file=label_file)
#features = feature(filepath="../MLDS_hw2_1_data/")
#train_list = get_training_list(filepath=filepath)
#print(train_list[0])
#annotated = annotate(filepath=filepath, label_file=label_file, word_dict=word_dict, w2i=w2i, i2w=i2w)
#print(annotated[45])