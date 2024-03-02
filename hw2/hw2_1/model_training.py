### import libraries
import numpy as np
import pandas as pd
import pickle
import random
from scipy.special import expit
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import json
import regex as re
import hw2.hw2_1.clean_txt as clean
import os
from collections import Counter
import time

### Build Dictionary
def build_dictionary(filepath, label_file):
    with open(filepath + label_file, "r") as f:
        data = json.load(f)

    word_counts = Counter()
    for entry in data:
        for caption in entry['caption']:
            word_counts.update(clean.clean_txt(caption, w2v=True).split())

    word_dict = {word: count for word, count in word_counts.items() if count > 4}
    useful_tokens = [(0,'<PAD>'), (1,'<BOS>'), (2,'<EOS>'), (3,'<UNK>')]

    i2w = {i + len(useful_tokens): word for i, word in enumerate(word_dict)}
    w2i = {word: i + len(useful_tokens) for i, word in enumerate(word_dict)}
    i2w.update(dict(useful_tokens))
    w2i.update({t[0]: i for i, t in enumerate(useful_tokens)})
    #w2i.update(dict(zip(t[0] for t in useful_tokens), range(len(useful_tokens))))

    return i2w, w2i, word_dict

### text to indices function
def text_to_indices(text, w2i):
    cleaned_text = clean.clean_txt(text, w2v=True)
    indices = [w2i.get(word, 3) for word in cleaned_text.split()]
    indices.insert(0, 1)  # Add BOS token
    indices.append(2)    # Add EOS token
    return indices

### annotation function
def annotate(filepath, label_file, w2i):
    with open(filepath + label_file, "r") as f:
        label = json.load(f)

    annotated_captions = []
    for entry in label:
        caption_indices = text_to_indices(entry["caption"], w2i)
        annotated_captions.append((entry["id"], caption_indices))

    return annotated_captions


# load the feature data 
def get_avi(filepath, files_dir):
    avi_data = {}
    training_feats = filepath + files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

### load and preprocess training dataimport torch
class TrainingData(Dataset):
    def __init__(self, filepath, label_file, files_dir, w2i, i2w=None):
        super().__init__()
        self.filepath = filepath
        self.label_file = label_file
        self.files_dir = files_dir
        self.w2i = w2i
        self.i2w = i2w  # Optional, unused in this class

        with open(filepath + label_file, "r") as f:
            label = json.load(f)

        self.data_pair = []
        for entry in label:
            for caption in entry["caption"]:  # Iterate through captions efficiently
                indices = text_to_indices(caption, w2i)
                self.data_pair.append((entry["id"], indices))

        self.avi = get_avi(self.filepath, self.files_dir)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.tensor(self.avi[avi_file_name])
        data += torch.randn_like(data) * 0.2  # More efficient noise addition
        return data, torch.tensor(sentence)



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
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state[0]
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context
    
# Encoder Class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    
        self.linear = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size() 
        input = input.view(-1, feat_n).clone().detach().type(torch.float32)
        input = self.linear(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, hidden_state = self.lstm(input)

        return output, hidden_state

# Decoder Class
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, word_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size=hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train'):
        batch_size, _ , _ = encoder_output.size()

        decoder_hidden = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_input = torch.ones(batch_size, 1).long().to(encoder_output.device)
        seq_logprobs, seq_predictions = [], []

        if targets is not None:  # Training mode
            targets = self.embedding(targets)
            _, seq_len , _ = targets.size()

            for i in range(seq_len - 1):
                teacher_forcing_ratio = 0.6 if mode == 'train' else 0.0  # Fixed teacher forcing ratio
                teacher_force = random.random() < teacher_forcing_ratio

                if teacher_force:
                    current_input = targets[:, i]
                else:
                    current_input = self.embedding(decoder_input).squeeze(1)

                context = self.attention(decoder_hidden, encoder_output)
                lstm_input = torch.cat([current_input, context], dim=1).unsqueeze(1)
                lstm_output, decoder_hidden = self.lstm(lstm_input, decoder_hidden)
                logprob = self.to_final_output(lstm_output.squeeze(1))
                seq_logprobs.append(logprob.unsqueeze(1))
                decoder_input = logprob.unsqueeze(1).max(2)[1]

        else:  # Inference mode
            seq_len = 28  # Fixed maximum output sequence length

            for i in range(seq_len - 1):
                current_input = self.embedding(decoder_input).squeeze(1)
                context = self.attention(decoder_hidden, encoder_output)
                lstm_input = torch.cat([current_input, context], dim=1).unsqueeze(1)
                lstm_output, decoder_hidden = self.lstm(lstm_input, decoder_hidden)
                logprob = self.to_final_output(lstm_output.squeeze(1))
                seq_logprobs.append(logprob.unsqueeze(1))
                decoder_input = logprob.unsqueeze(1).max(2)[1]

        seq_logprobs = torch.cat(seq_logprobs, dim=1)
        seq_predictions = seq_logprobs.max(2)[1]
        return seq_logprobs, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        return self.forward(encoder_last_hidden_state, encoder_output, mode='infer')

### Model class
class ModelClass(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        model_output = self.decoder(
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_output=encoder_outputs,
            targets=target_sentences,
            mode=mode
        )

        return model_output

### loss function
def calculate_loss(loss_fn, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] -1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss/batch_size

    return loss

### train
def train(model, epoch, loss_fn, optimizer, train_loader, device):
    model.train()
    print(epoch)
    
    for _, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats.to(device), ground_truths.to(device)
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)
        
        optimizer.zero_grad()
        seq_logProb,_ = model(avi_feats, target_sentences = ground_truths, mode = 'train')
        ground_truths = ground_truths[:, 1:]  
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)
        
        loss.backward()
        optimizer.step()

    loss = loss.item()
    print(loss)


def main():
    filepath = "../MLDS_hw2_1_data/"
    files_dir = 'training_data/feat'
    label_file = 'training_label.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i2w, w2i, word_dict = build_dictionary(filepath=filepath, label_file=label_file)
    with open('i2w.pkl', 'wb') as f:
        pickle.dump(i2w, f)
    train_dataset = TrainingData(filepath=filepath, label_file=label_file, files_dir=files_dir,
                                 w2i=w2i, i2w=i2w)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)
    
    epochs_n = 100

    encoder = Encoder()
    decoder = DecoderWithAttention(512, len(i2w) +4, len(i2w) +4, 1024, 0.3)
    model = ModelClass(encoder=encoder, decoder=decoder)
    model = nn.DataParallel(model)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.0001)
    
    start = time.time()
    for epoch in range(epochs_n):
        train(model, epoch+1, loss_fn, optimizer, train_dataloader, device) 

    end = time.time()
    torch.save(model, "{}.h5".format('modelv1_seq2seq'))
    print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))

if __name__ == "__main__":
    main()