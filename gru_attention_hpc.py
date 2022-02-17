#!/home/kchandrasekaran/virtualenvs/pytorch/bin/python
#SBATCH --job-name=GRU_attention
#SBATCH --output=GRU_activities_attention.log
#SBATCH --mem=32G
#SBATCH -n 8
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH -C V100
#SBATCH -p emmanuel

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as met
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from scipy import signal
import itertools
from collections import deque
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parameter

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import TimeSeriesSplit



X_train = np.loadtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/Train/X_train.txt")
X_test = np.loadtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/Test/X_test.txt")
Y_train = np.loadtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/Train/y_train.txt")
Y_test = np.loadtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/Test/y_test.txt")

X = np.vstack((X_train, X_test))
Y = np.hstack((Y_train, Y_test))

feature_names = np.genfromtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/features.txt",dtype='str')
label_names = np.genfromtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/activity_labels.txt", dtype='str')
label_names = [l[1] for l in label_names ]
label_names

transition_label_names=np.array(['ACTIVITIES', 'STAND_TO_SIT',
 'SIT_TO_STAND',
 'SIT_TO_LIE',
 'LIE_TO_SIT',
 'STAND_TO_LIE',
 'LIE_TO_STAND'])

activity_label_names=np.array(['WALKING',
 'WALKING_UPSTAIRS',
 'WALKING_DOWNSTAIRS',
 'SITTING',
 'STANDING',
 'LAYING',
'TRANSITIONS'])

X_transition_train=X[:int(len(X)*0.6)]
X_transition_validation=X[int(len(X)*0.6):int(len(X)*0.8)]
X_transition_test=X[int(len(X)*0.8):]
Y_transition_train=np.where(Y[:int(len(Y)*0.6)] > 6, Y[:int(len(Y)*0.6)], 0)
Y_transition_validation=np.where(Y[int(len(Y)*0.6):int(len(X)*0.8)] > 6, Y[int(len(Y)*0.6):int(len(X)*0.8)], 0)
Y_transition_test=np.where(Y[int(len(Y)*0.8):] > 6, Y[int(len(Y)*0.8):], 0)

X_activities_train=X[:int(len(X)*0.6)]
X_activities_validation=X[int(len(X)*0.6):int(len(X)*0.8)]
X_activities_test=X[int(len(X)*0.8):]
Y_activities_train=np.where(Y[:int(len(Y)*0.6)] < 7, Y[:int(len(Y)*0.6)], 0)
Y_activities_validation=np.where(Y[int(len(Y)*0.6):int(len(X)*0.8)] < 7, Y[int(len(Y)*0.6):int(len(X)*0.8)], 0)
Y_activities_test=np.where(Y[int(len(Y)*0.8):] < 7, Y[int(len(Y)*0.8):], 0)

train_uid = np.loadtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/Train/subject_id_train.txt")
test_uid = np.loadtxt("/home/kchandrasekaran/wash/Datasets/uci-smartphone-based-recognition-of-human-activities/original/Test/subject_id_test.txt")
user_ids = np.hstack((train_uid, test_uid))
rand_uid=[np.random.choice(np.unique(user_ids), len(np.unique(user_ids)), replace=False) for _ in range(5)]

bidirectional = True
#bidirectional = False
if bidirectional:
    num_directions = 2
else:
    num_directions = 1

def one_hot(y, labels):
    Y_onehot=[]
    for l in y:
        empty_label=np.zeros(len(labels))
        empty_label[labels.index(l)]=1.
        Y_onehot.append(empty_label)
    return(np.vstack(Y_onehot))


def get_metrics(target, output):
        
    pred = np.round(output)

    tp = np.sum(((pred + target) == 2).astype(float), axis=0)
    fp = np.sum(((pred - target) == 1).astype(float), axis=0)
    fn = np.sum(((pred - target) == -1).astype(float), axis=0)
    tn = np.sum(((pred + target) == 0).astype(float), axis=0)

    acc = (tp + tn) / (tp + tn + fp + fn)
    try:
        prec = tp / (tp + fp)
    except ZeroDivisionError:
        prec = 0.0
    try:
        rec = tp / (tp + fn)
    except ZeroDivisionError:
        rec = 0.0
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0.0


    try:
        f1=2.*((prec*rec)/(prec+rec))
    except ZeroDivisionError:
        f1 = 0.0

    acc[acc != acc] = 0.
    prec[prec != prec] = 0.
    rec[rec != rec] = 0.
    specificity[specificity != specificity] = 0.
    f1[f1 != f1] = 0.

    balanced_accuracy = (rec + specificity) / 2.

    err_rate = np.subtract(1., acc)
    f1_micro, f1_macro, f1_weight, log_ls, roc = [], [], [], [], []
    for idx in range(target.shape[1]):
        y_test=target[:,idx]
        y_pred=pred[:,idx]

        f1_micro.append(f1_score(y_test, y_pred, average= 'micro'))
        f1_macro.append(f1_score(y_test, y_pred, average= 'macro'))
        f1_weight.append(f1_score(y_test, y_pred, average= 'weighted'))
        log_ls.append(log_loss(y_test, y_pred, labels=[0., 1.]))
        try:
            roc.append(roc_auc_score(y_test, output[:, idx]))
        except ValueError:
            roc.append(np.nan)

    return (balanced_accuracy, acc, err_rate, prec, rec, specificity, f1, tp, fp, fn, tn, np.array(f1_micro), np.array(f1_macro), np.array(f1_weight), np.array(log_ls), np.array(roc))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.cuda.FloatTensor
else:
    device = torch.device("cpu")
    dtype = torch.FloatTensor

def new_parameter(*size):
    out =nn.Parameter(dtype(*size))
    torch.nn.init.xavier_normal(out)
    return out

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 1).unsqueeze(0)

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size, attention_size):
        super(CustomRNN, self).__init__()
        self.output_size=output_size
        self.rnn = nn.GRU(input_size=input_size, num_layers= num_layers, hidden_size=hidden_size, 
                          batch_first=True, bidirectional=bidirectional, dropout=0.1).cuda()
        self.linear = nn.Linear(hidden_size*num_directions, output_size)
        self.weights = new_parameter(hidden_size*num_directions, hidden_size*num_directions)
        self.attention = new_parameter(attention_size*num_directions, 1)
        self.act = nn.Softmax()
        #self.act = nn.Tanh()
    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        #print(pred.shape, self.attention.shape)
        #pred = self.act(self.linear(pred)).squeeze().cuda()
        #print(hidden.shape)
        #attention_score = torch.matmul(hidden, self.attention).squeeze()
        squish = batch_matmul(pred, self.weights, nonlinearity='tanh')
        #print(squish.unsqueeze(dim=1).shape)
        attention_score = batch_matmul(squish.unsqueeze(dim=1), self.attention)
        #print(attention_score.shape)
        #attention_score = torch.squeeze(F.softmax(attention_score))
        #attention_score = F.softmax(attention_score).view(hidden.size(0), hidden.size(1), 1)
        attention_normalized = F.softmax(attention_score)
        #print(attention_normalized.shape)
        scored_x = attention_mul(pred, attention_normalized)
        #scored_x = hidden * attention_score
        #print(scored_x.shape)
        # now, sum across dim 1 to get the expected feature vector
        #condensed_x = torch.sum(scored_x, dim=2)
        #print(attention_score.shape)
        #print(condensed_x.shape)
        #print(attention_score.detach().cpu().numpy()[0].shape)
        #plt.matshow(condensed_x.detach().cpu().numpy())
        #print(pred.shape)
        #print(torch.t(condensed_x).expand_as(pred).shape)
        #t1 = torch.matmul(pred.view(pred.data.shape[0], pred.data.shape[2]).cuda(), condensed_x)
        #t1 = torch.mul(pred , torch.t(condensed_x).expand_as(pred))
        #print(scored_x.shape)
        t2=self.linear(scored_x.squeeze(0))
        #print(t2.shape)
        pred = self.act(t2)
        #print(pred.shape)
        return pred
    
def train_gru(config, train_inp, train_out, test_inp, test_out):
    r= CustomRNN(config["input_dim"], config["hidden_size"], config["num_layers"], config["output_dim"], config["hidden_size"]).to(device)
    #print([p for p in r.parameters()])
    predictions = []
    optimizer = torch.optim.Adam(r.parameters(), lr=config["learning_rate"])
    loss_func = F.binary_cross_entropy
    inp = Variable(torch.from_numpy(train_inp.reshape((train_inp.shape[0], -1, config["input_dim"]))).type(dtype), requires_grad=True)
    out = Variable(torch.from_numpy(train_out.reshape((train_inp.shape[0], config["output_dim"]))).type(dtype))
    for t in range(config["num_epochs"]):
        hidden = None
        inp = Variable(torch.from_numpy(train_inp.reshape((train_inp.shape[0], -1, config["input_dim"]))).type(dtype), requires_grad=True)
        out = Variable(torch.from_numpy(train_out.reshape((train_inp.shape[0], config["output_dim"]))).type(dtype))

        pred = r(inp)
        optimizer.zero_grad()
        predictions.append(pred.data.cpu().numpy())
        loss = loss_func(pred, out)
        if t%100==0:
            print(t, loss.item())
        loss.backward()
        optimizer.step()

    t_inp = Variable(torch.Tensor(test_inp.reshape((test_inp.shape[0], -1, 561))).type(dtype), requires_grad=True)
    pred_t = r(t_inp)
    pred_one_hot = one_hot(np.argmax(pred_t.data.cpu().numpy(), axis=1)+1, range(1, len(transition_label_names)+1))
    results_metrics = get_metrics(test_out, pred_one_hot)
    results_conf_mat = confusion_matrix(np.argmax(test_out, axis=1)+1, np.argmax(pred_t.data.cpu().numpy(), axis=1)+1)
    metric_names = np.array(["CV", "Balanced Accuracy", "Accuracy", "Error Rate","Precision","Recall","Specificity", "F1", "TP","FP","FN","TN", "Micro F1","Macro F1","Weighted F1","Log-Loss","ROC AUC"])
    results = np.hstack((metric_names.reshape(-1, 1), np.vstack((transition_label_names, np.vstack(results_metrics)))))
    
    filename = config["result_filename"]
    
    with open("{}.csv".format(filename), 'a') as f:
        pd.DataFrame(results).to_csv(f, header=False)
    with open("{}_conf_matrix.csv".format(filename), 'a') as f:
        pd.DataFrame(np.hstack((transition_label_names.reshape(-1, 1), results_conf_mat))).to_csv(f, header=False)
    
    filename_modifier = 0
    while(os.path.exists("{}_predictions_probabilities_target_{}".format(filename, filename_modifier)+".csv")):
        filename_modifier+=1
    pred_fname="{}_predictions_probabilities_target_{}".format(filename, filename_modifier)+".csv"
    
    with open(pred_fname, 'a') as f:
        predictions=np.vstack((list(transition_label_names)*3,np.hstack((test_out, torch.round(pred_t.cpu().numpy()),pred_t.cpu().numpy()))))
        pd.DataFrame(predictions, columns=["Y"]*(len(transition_label_names))+["Y_Hat"]*(len(transition_label_names))+["Y_Hat_Prob"]*(len(transition_label_names))).to_csv(f, index=False)
       
    
    plt.subplots(figsize=(20,15))
    s=sns.heatmap(results_conf_mat.astype(int), annot=True, annot_kws={"size": 20}, cmap="YlGnBu", fmt='d', xticklabels=transition_label_names, yticklabels=transition_label_names)
    title="Transition Learning"
    s.set_title(title)
    
    filename_modifier = 0
    while(os.path.exists(filename+"_"+str(filename_modifier)+".png")):
        filename_modifier+=1
    fig_fname=filename+"_"+str(filename_modifier)+".png"
    s.get_figure().savefig(fig_fname, dpi=400)
    

config = {}
config["input_dim"] = 561
config["hidden_size"] = 512
config["num_layers"] = 2
config["num_directions"] = 2
config["output_dim"] = 7
config["num_epochs"] = 1000
config["learning_rate"] = 1e-3
config["result_filename"] = "results/Transition_classification_GRU_w_attention_{}_directions_{}_layers_{}_lr_{}_units_{}_epochs".format(config["num_directions"], config["num_layers"], config["learning_rate"], config["hidden_size"], config["num_epochs"])

for cv_idx, cv_fold in enumerate(rand_uid):
    train_ids, val_ids, test_ids = cv_fold[:int(0.6*len(cv_fold))], cv_fold[int(0.6*len(cv_fold)):int(0.8*len(cv_fold)):], cv_fold[int(0.8*len(cv_fold)):]

    train_idx = np.isin(user_ids, train_ids)
    val_idx = np.isin(user_ids, val_ids)
    test_idx = np.isin(user_ids, test_ids)
    
    X_transition_train=X[train_idx]
    X_transition_validation=X[val_idx]
    X_transition_test=X[test_idx]
    Y_transition_train=np.where(Y[train_idx] > 6, Y[train_idx], 0)
    Y_transition_validation=np.where(Y[val_idx] > 6, Y[val_idx], 0)
    Y_transition_test=np.where(Y[test_idx] > 6, Y[test_idx], 0)

    train_inp, train_out = X_transition_train, one_hot(Y_transition_train, [0,7,8,9,10,11,12])
    val_inp, val_out = X_transition_validation, one_hot(Y_transition_validation, [0,7,8,9,10,11,12])
    test_inp, test_out = X_transition_test, one_hot(Y_transition_test, [0,7,8,9,10,11,12])
        
    train_gru(config, train_inp, train_out, val_inp, val_out)
    
