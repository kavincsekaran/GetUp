#!/home/kchandrasekaran/virtualenvs/pytorch/bin/python
#SBATCH --job-name=Tune_BiGRU_inverse_GridSearch
#SBATCH --output=Tune_BiGRU_inverse_GridSearch.log
#SBATCH --mem=32G
#SBATCH -n 32
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH -C V100
#SBATCH -p emmanuel


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import itertools
from collections import deque
from collections import Counter
import os
import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.svm import SVC
from xgboost import XGBClassifier

import sklearn.metrics as met
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
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

bidirectional = True
if bidirectional:
    num_directions = 2
else:
    num_directions = 1

class Custom_GRU(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(Custom_GRU, self).__init__()
        self.output_size=output_size
        self.rnn = nn.GRU(input_size=input_size, num_layers= num_layers, hidden_size=hidden_size, 
                          batch_first=True, bidirectional=bidirectional, dropout=0.1).cuda()
        self.linear = nn.Linear(hidden_size*num_directions, output_size)
        self.act = nn.Softmax()
        #self.act = nn.Tanh()
    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        pred = self.act(self.linear(pred)).view(pred.data.shape[0], self.output_size).cuda()
        #pred = nn.Sigmoid(self.linear(pred)).view(pred.data.shape[0], 12)
        return pred

def train_gru(config, train_inp, train_out, test_inp, test_out):
    r= Custom_GRU(config["input_dim"], config["hidden_size"], config["num_layers"], config["output_dim"]).to(device)
    predictions = []
    optimizer = torch.optim.Adam(r.parameters(), lr=config["learning_rate"])
    #loss_func = nn.L1Loss()
    loss_func = F.binary_cross_entropy
    train_loss = [] 
    for t in range(config["num_epochs"]):
        hidden = None
        inp = Variable(torch.from_numpy(train_inp.reshape((train_inp.shape[0], -1, config["input_dim"]))).type(dtype), requires_grad=True)
        out = Variable(torch.from_numpy(train_out.reshape((train_inp.shape[0], config["output_dim"]))).type(dtype))
    
        pred = r(inp)
        optimizer.zero_grad()
        predictions.append(pred.data.cpu().numpy())
        loss = loss_func(pred, out)
        train_loss.append(loss)
        #if t%100==0:
        #    print(t, loss.data[0])
        loss.backward()
        optimizer.step()
        
    t_inp = Variable(torch.Tensor(test_inp.reshape((test_inp.shape[0], -1, config["input_dim"]))).type(dtype), requires_grad=True)
    t_out = Variable(torch.Tensor(test_out.reshape((test_out.shape[0], config["output_dim"]))).type(dtype))
    pred_t = r(t_inp)
    test_loss = loss_func(pred_t, t_out)
    pred_numpy = pred_t.data.cpu().numpy()
    transition_act=np.where(np.argmax(pred_numpy, axis=1)==6, True, False)
    print(transition_act)
    groups = [(list(v), g) for g,v in itertools.groupby(transition_act)]
    processed=0
    transition_y_hat=[]
    #print(groups)
    for group in groups:
        cont=group[0]
        g=group[1]
        trans_pred=0
        if g:
            if(len(cont)>0):
                prev_act=np.argmax(pred_numpy[processed-1])+1
                next_act=np.argmax(pred_numpy[processed+len(cont)])+1
                if(prev_act==5 and next_act==4):
                    trans_pred=7
                elif(prev_act==5 and next_act==6):
                    trans_pred=11
                elif(prev_act==4 and next_act==5):
                    trans_pred=8
                elif(prev_act==4 and next_act==6):
                    trans_pred=9
                elif(prev_act==6 and next_act==4):
                    trans_pred=10
                elif(prev_act==6 and next_act==5):
                    #print("class 12 should be predicted")
                    trans_pred=12
                elif(prev_act==6 and next_act==1):
                    trans_pred=12
                else:
                    print("Unexpected Combination. Prev{}. Next{}".format(prev_act, next_act))
                processed+=len(cont)
            else:
                processed+=len(cont)
        else:
            processed+=len(cont)
        transition_y_hat.append(np.ones_like(cont)*trans_pred)
    transition_y_hat = np.hstack(transition_y_hat)
    
    pred_one_hot = one_hot(transition_y_hat, [0, 7, 8, 9, 10, 11, 12])
    
    results_metrics = get_metrics(test_out, pred_one_hot)
    results_conf_mat = confusion_matrix(np.argmax(test_out, axis=1)+1, np.argmax(pred_one_hot, axis=1)+1)
    metric_names = np.array(["CV", "Balanced Accuracy", "Accuracy", "Error Rate", "Precision","Recall","Specificity", "F1", "TP","FP","FN","TN", "Micro F1","Macro F1","Weighted F1","Log-Loss","ROC AUC"])
    results = np.hstack((metric_names.reshape(-1, 1), np.vstack((transition_label_names, np.vstack(results_metrics)))))
    
    filename= config["result_filename"]
    with open("{}.csv".format(filename), 'a') as f:
        pd.DataFrame(results).to_csv(f, header=False)
    with open("{}_conf_matrix.csv".format(filename), 'a') as f:    
        pd.DataFrame(np.hstack((transition_label_names.reshape(-1, 1), results_conf_mat))).to_csv(f, header=False)
    plt.subplots(figsize=(20,15))
    sns.set(font_scale = 1.8)
    s=sns.heatmap(results_conf_mat.astype(int), annot=True, annot_kws={"size": 20}, cmap="YlGnBu", fmt='d', xticklabels=transition_label_names, yticklabels=transition_label_names)
    title="Transition Learning"
    s.set_title(title)
    
    filename_modifier = 0
    while(os.path.exists(filename+"_"+str(filename_modifier)+".png")):
        filename_modifier+=1
    fig_fname=filename+"_"+str(filename_modifier)+".png"
    s.get_figure().savefig(fig_fname, dpi=400)
    
    with open(filename+"_"+str(filename_modifier)+"_train_loss.csv", 'wb') as train_loss_file:
        wr = csv.writer(train_loss_file, quoting=csv.QUOTE_ALL)
        wr.writerow(train_loss)
    
    return test_loss
    

hidden_layer_sizes=[256, 512, 1024]
learning_rates=[1e-2, 1e-3, 1e-4, 1e-5]
epochs=[100, 500, 1000, 5000]
directions = [1, 2]
layers=[1, 2, 3, 4]
    
hyperparameters = [hidden_layer_sizes, directions, layers, epochs, learning_rates]
all_parameter_combinations=list(itertools.product(*hyperparameters))
costs=[]
for parameter_combo in all_parameter_combinations:
    config = {}
    config["input_dim"] = 561
    config["hidden_size"] = parameter_combo[0] 
    config["num_directions"] = parameter_combo[1]
    config["num_layers"] = parameter_combo[2]
    config["output_dim"] = 7
    config["num_epochs"] = parameter_combo[3]
    config["learning_rate"] = parameter_combo[4]
    config["result_filename"] = "Inverse_Transition_classification_results_BiGRU_{}_directions_{}_layers_{}_lr_{}_units_{}_batch_size_{}_epochs".format(config["num_directions"], config["num_layers"], config["learning_rate"], config["hidden_size"],config["batch_size"], config["num_epochs"])
    for cv_idx, cv_fold in enumerate(rand_uid):
        train_ids, val_ids, test_ids = cv_fold[:int(0.6*len(cv_fold))], cv_fold[int(0.6*len(cv_fold)):int(0.8*len(cv_fold)):], cv_fold[int(0.8*len(cv_fold)):]
        train_idx = np.isin(user_ids, train_ids)
        val_idx = np.isin(user_ids, val_ids)
        test_idx = np.isin(user_ids, test_ids)
        
        X_activity_train=X[train_idx]
        X_activity_validation=X[val_idx]
        X_activity_test=X[test_idx]
        Y_activity_train=np.where(Y[train_idx] <= 6, Y[train_idx], 0)
        Y_activity_validation=np.where(Y[val_idx] <= 6, Y[val_idx], 0)
        Y_activity_test=np.where(Y[test_idx] <= 6, Y[test_idx], 0)
        
        X_transition_train=X[train_idx]
        X_transition_validation=X[val_idx]
        X_transition_test=X[test_idx]
        Y_transition_train=np.where(Y[train_idx] > 6, Y[train_idx], 0)
        Y_transition_validation=np.where(Y[val_idx] > 6, Y[val_idx], 0)
        Y_transition_test=np.where(Y[test_idx] > 6, Y[test_idx], 0)
        
        train_act_inp, train_act_out = X_activity_train, one_hot(Y_activity_train, [1, 2, 3, 4, 5, 6, 0])
        val_act_inp, val_act_out = X_activity_validation, one_hot(Y_activity_validation, [1, 2, 3, 4, 5, 6, 0])
        test_act_inp, test_act_out = X_activity_test, one_hot(Y_activity_test, [1, 2, 3, 4, 5, 6, 0])
            
        train_trans_inp, train_trans_out = X_transition_train, one_hot(Y_transition_train, [0,7,8,9,10,11,12])
        val_trans_inp, val_trans_out = X_transition_validation, one_hot(Y_transition_validation, [0,7,8,9,10,11,12])
        test_trans_inp, test_trans_out = X_transition_test, one_hot(Y_transition_test, [0,7,8,9,10,11,12])
        start_time = time.time()
        loss = train_gru(config, train_act_inp, train_act_out, val_act_inp, val_trans_out)
        costs.append(loss)
        config["CV_index"] = cv_idx
        config["test_loss"] = loss
        config["time_elapsed"] = time.time()-start_time
        with open("GRU_Inverse_GridSearch_results.csv".format(filename), 'a') as f:    
            pd.DataFrame.from_dict(config).to_csv(f)
                
        best_params=all_parameter_combinations[np.argmin(costs)]
                
