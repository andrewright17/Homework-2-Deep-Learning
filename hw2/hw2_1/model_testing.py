import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
import json
from torch.utils.data import DataLoader, Dataset
from hw2.hw2_1.bleu_eval import BLEU
from hw2.hw2_1.model_training import *
import pickle

class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]
    
def testing( model, test_loader, i2w, device):
    model.eval()
    ss = []
    for _, batch in enumerate(test_loader):
        id, avi_feats = batch
        avi_feats = avi_feats.to(device)
        id, avi_feats = id, Variable(avi_feats).float()

        _, seq_predictions = model(avi_feat = avi_feats, mode='infer')
        test_predictions = seq_predictions
        
        result = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(id, result)
        for r in rr:
            ss.append(r)
    return ss

def main():
    data_dir = '{}'.format(sys.argv[1])
    dataset = test_data(data_dir + '/feat')
    testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    with open('i2w.pkl', 'rb') as f:
        i2w = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('modelv1_seq2seq.h5')
    model = model.to(device)

    preds = testing(model, test_loader = testing_loader, i2w = i2w, device = device)
    with open(sys.argv[2], 'w') as f:
        for id, cap in preds:
            f.write('{},{}\n'.format(id, cap))
    test = json.load(open(data_dir + '/testing_label.json'))
    output = sys.argv[2]
    result = {}
    with open(output,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))
    #print(result['04Gt01vatkk_248_265.avi'])
    #print([item['caption'] for item in test if item['id'] == '04Gt01vatkk_248_265.avi'])

if __name__ == "__main__" :
    main()