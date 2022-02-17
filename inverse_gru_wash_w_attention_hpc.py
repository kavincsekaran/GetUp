#!/home/kchandrasekaran/virtualenvs/python3/bin/python
#SBATCH --job-name=BiGRU_Wash
#SBATCH --output=BiGRU_Wash.log
#SBATCH --mem=32G
#SBATCH -n 2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH -C V100
#SBATCH -p emmanuel



import time
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
import pickle

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

label_titles_clean = ['Walking',
                      'Lying Down',
                      'Standing',
                      'Phone in Bag',
                      'Phone in Hand',
                      'Phone in Pocket',
                      'Phone on Table',
                      'Sitting',
                      'Sleeping',
                      'Talking',
                      'Toilet']

sensor_titles_clean = ['Accelerometer',
                       'Gyroscope',
                       'Magnetometer',
                       'Location Services',
                       'Audio',
                       'Discrete',
                       'Other']

watch_features = ['watch_acceleration:magnitude_stats:mean',
                  'watch_acceleration:magnitude_stats:std',
                  'watch_acceleration:magnitude_stats:moment3',
                  'watch_acceleration:magnitude_stats:moment4',
                  'watch_acceleration:magnitude_stats:percentile25',
                  'watch_acceleration:magnitude_stats:percentile50',
                  'watch_acceleration:magnitude_stats:percentile75',
                  'watch_acceleration:magnitude_stats:value_entropy',
                  'watch_acceleration:magnitude_stats:time_entropy',
                  'watch_acceleration:magnitude_spectrum:log_energy_band0',
                  'watch_acceleration:magnitude_spectrum:log_energy_band1',
                  'watch_acceleration:magnitude_spectrum:log_energy_band2',
                  'watch_acceleration:magnitude_spectrum:log_energy_band3',
                  'watch_acceleration:magnitude_spectrum:log_energy_band4',
                  'watch_acceleration:magnitude_spectrum:spectral_entropy',
                  'watch_acceleration:magnitude_autocorrelation:period',
                  'watch_acceleration:magnitude_autocorrelation:normalized_ac',
                  'watch_acceleration:3d:mean_x',
                  'watch_acceleration:3d:mean_y',
                  'watch_acceleration:3d:mean_z',
                  'watch_acceleration:3d:std_x',
                  'watch_acceleration:3d:std_y',
                  'watch_acceleration:3d:std_z',
                  'watch_acceleration:3d:ro_xy',
                  'watch_acceleration:3d:ro_xz',
                  'watch_acceleration:3d:ro_yz',
                  'watch_acceleration:spectrum:x_log_energy_band0',
                  'watch_acceleration:spectrum:x_log_energy_band1',
                  'watch_acceleration:spectrum:x_log_energy_band2',
                  'watch_acceleration:spectrum:x_log_energy_band3',
                  'watch_acceleration:spectrum:x_log_energy_band4',
                  'watch_acceleration:spectrum:y_log_energy_band0',
                  'watch_acceleration:spectrum:y_log_energy_band1',
                  'watch_acceleration:spectrum:y_log_energy_band2',
                  'watch_acceleration:spectrum:y_log_energy_band3',
                  'watch_acceleration:spectrum:y_log_energy_band4',
                  'watch_acceleration:spectrum:z_log_energy_band0',
                  'watch_acceleration:spectrum:z_log_energy_band1',
                  'watch_acceleration:spectrum:z_log_energy_band2',
                  'watch_acceleration:spectrum:z_log_energy_band3',
                  'watch_acceleration:spectrum:z_log_energy_band4',
                  'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0',
                  'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1',
                  'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2',
                  'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3',
                  'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4',
                  'watch_heading:mean_cos',
                  'watch_heading:std_cos',
                  'watch_heading:mom3_cos',
                  'watch_heading:mom4_cos',
                  'watch_heading:mean_sin',
                  'watch_heading:std_sin',
                  'watch_heading:mom3_sin',
                  'watch_heading:mom4_sin',
                  'watch_heading:entropy_8bins']

contexts_other = ['label:LAB_WORK',
                  'label:IN_CLASS',
                  'label:IN_A_MEETING',
                  'label:LOC_main_workplace',
                  'label:ON_A_BUS',
                  'label:DRIVE_-_I_M_THE_DRIVER',
                  'label:DRIVE_-_I_M_A_PASSENGER',
                  'label:LOC_home',
                  'label:FIX_restaurant',
                  'label:COOKING',
                  'label:SHOPPING',
                  'label:STROLLING',
                  'label:DRINKING__ALCOHOL_',
                  'label:BATHING_-_SHOWER',
                  'label:CLEANING',
                  'label:DOING_LAUNDRY',
                  'label:WASHING_DISHES',
                  'label:WATCHING_TV',
                  'label:AT_A_PARTY',
                  'label:AT_A_BAR',
                  'label:LOC_beach',
                  'label:COMPUTER_WORK',
                  'label:EATING',
                  'label:DRESSING',
                  'label:AT_THE_GYM',
                  'label:AT_SCHOOL',
                  'label:WITH_CO-WORKERS',
                  'label:WITH_FRIENDS',
                  'label_source']

contexts_activity = ['label:LYING_DOWN',
                     'label:SITTING',
                     'label:FIX_walking',
                     'label:FIX_running',
                     'label:BICYCLING',
                     'label:SLEEPING',
                     'label:IN_A_CAR',
                     'label:OR_exercise',
                     'label:SURFING_THE_INTERNET',
                     'label:TALKING',
                     'label:EATING',
                     'label:TOILET',
                     'label:GROOMING',
                     'label:STAIRS_-_GOING_UP',
                     'label:STAIRS_-_GOING_DOWN',
                     'label:ELEVATOR',
                     'label:OR_standing']

contexts_in_out = ['label:OR_indoors',
                   'label:OR_outside']

contexts_prioception = ['label:PHONE_IN_POCKET',
                        'label:PHONE_IN_HAND',
                        'label:PHONE_IN_BAG',
                        'label:PHONE_ON_TABLE']

contexts_immediate = ['label:LYING_DOWN',
                      'label:SITTING',
                      'label:FIX_walking',
                      'label:SLEEPING',
                      'label:OR_standing',
                      'label:PHONE_IN_POCKET',
                      'label:PHONE_IN_HAND',
                      'label:PHONE_IN_BAG',
                      'label:PHONE_ON_TABLE',
                      'label:TOILET',
                      'label:TALKING']

label_information = ['timestamp',
                     'label_source']

accelerometer = ['raw_acc:magnitude_stats:mean',
                 'raw_acc:magnitude_stats:std',
                 'raw_acc:magnitude_stats:moment3',
                 'raw_acc:magnitude_stats:moment4',
                 'raw_acc:magnitude_stats:percentile25',
                 'raw_acc:magnitude_stats:percentile50',
                 'raw_acc:magnitude_stats:percentile75',
                 'raw_acc:magnitude_stats:value_entropy',
                 'raw_acc:magnitude_stats:time_entropy',
                 'raw_acc:magnitude_spectrum:log_energy_band0',
                 'raw_acc:magnitude_spectrum:log_energy_band1',
                 'raw_acc:magnitude_spectrum:log_energy_band2',
                 'raw_acc:magnitude_spectrum:log_energy_band3',
                 'raw_acc:magnitude_spectrum:log_energy_band4',
                 'raw_acc:magnitude_spectrum:spectral_entropy',
                 'raw_acc:magnitude_autocorrelation:period',
                 'raw_acc:magnitude_autocorrelation:normalized_ac',
                 'raw_acc:3d:mean_x',
                 'raw_acc:3d:mean_y',
                 'raw_acc:3d:mean_z',
                 'raw_acc:3d:std_x',
                 'raw_acc:3d:std_y',
                 'raw_acc:3d:std_z',
                 'raw_acc:3d:ro_xy',
                 'raw_acc:3d:ro_xz',
                 'raw_acc:3d:ro_yz']

gyroscope = ['proc_gyro:magnitude_stats:mean',
             'proc_gyro:magnitude_stats:std',
             'proc_gyro:magnitude_stats:moment3',
             'proc_gyro:magnitude_stats:moment4',
             'proc_gyro:magnitude_stats:percentile25',
             'proc_gyro:magnitude_stats:percentile50',
             'proc_gyro:magnitude_stats:percentile75',
             'proc_gyro:magnitude_stats:value_entropy',
             'proc_gyro:magnitude_stats:time_entropy',
             'proc_gyro:magnitude_spectrum:log_energy_band0',
             'proc_gyro:magnitude_spectrum:log_energy_band1',
             'proc_gyro:magnitude_spectrum:log_energy_band2',
             'proc_gyro:magnitude_spectrum:log_energy_band3',
             'proc_gyro:magnitude_spectrum:log_energy_band4',
             'proc_gyro:magnitude_spectrum:spectral_entropy',
             'proc_gyro:magnitude_autocorrelation:period',
             'proc_gyro:magnitude_autocorrelation:normalized_ac',
             'proc_gyro:3d:mean_x',
             'proc_gyro:3d:mean_y',
             'proc_gyro:3d:mean_z',
             'proc_gyro:3d:std_x',
             'proc_gyro:3d:std_y',
             'proc_gyro:3d:std_z',
             'proc_gyro:3d:ro_xy',
             'proc_gyro:3d:ro_xz',
             'proc_gyro:3d:ro_yz']

magnetometer = ['raw_magnet:magnitude_stats:mean',
                'raw_magnet:magnitude_stats:std',
                'raw_magnet:magnitude_stats:moment3',
                'raw_magnet:magnitude_stats:moment4',
                'raw_magnet:magnitude_stats:percentile25',
                'raw_magnet:magnitude_stats:percentile50',
                'raw_magnet:magnitude_stats:percentile75',
                'raw_magnet:magnitude_stats:value_entropy',
                'raw_magnet:magnitude_stats:time_entropy',
                'raw_magnet:magnitude_spectrum:log_energy_band0',
                'raw_magnet:magnitude_spectrum:log_energy_band1',
                'raw_magnet:magnitude_spectrum:log_energy_band2',
                'raw_magnet:magnitude_spectrum:log_energy_band3',
                'raw_magnet:magnitude_spectrum:log_energy_band4',
                'raw_magnet:magnitude_spectrum:spectral_entropy',
                'raw_magnet:magnitude_autocorrelation:period',
                'raw_magnet:magnitude_autocorrelation:normalized_ac',
                'raw_magnet:3d:mean_x',
                'raw_magnet:3d:mean_y',
                'raw_magnet:3d:mean_z',
                'raw_magnet:3d:std_x',
                'raw_magnet:3d:std_y',
                'raw_magnet:3d:std_z',
                'raw_magnet:3d:ro_xy',
                'raw_magnet:3d:ro_xz',
                'raw_magnet:3d:ro_yz',
                'raw_magnet:avr_cosine_similarity_lag_range0',
                'raw_magnet:avr_cosine_similarity_lag_range1',
                'raw_magnet:avr_cosine_similarity_lag_range2',
                'raw_magnet:avr_cosine_similarity_lag_range3',
                'raw_magnet:avr_cosine_similarity_lag_range4']

location = ['location:num_valid_updates',
            'location:log_latitude_range',
            'location:log_longitude_range',
            'location:min_altitude',
            'location:max_altitude',
            'location:min_speed',
            'location:max_speed',
            'location:best_horizontal_accuracy',
            'location:best_vertical_accuracy',
            'location:diameter',
            'location:log_diameter',
            'location_quick_features:std_lat',
            'location_quick_features:std_long',
            'location_quick_features:lat_change',
            'location_quick_features:long_change',
            'location_quick_features:mean_abs_lat_deriv',
            'location_quick_features:mean_abs_long_deriv']

audio = ['audio_naive:mfcc0:mean',
         'audio_naive:mfcc1:mean',
         'audio_naive:mfcc2:mean',
         'audio_naive:mfcc3:mean',
         'audio_naive:mfcc4:mean',
         'audio_naive:mfcc5:mean',
         'audio_naive:mfcc6:mean',
         'audio_naive:mfcc7:mean',
         'audio_naive:mfcc8:mean',
         'audio_naive:mfcc9:mean',
         'audio_naive:mfcc10:mean',
         'audio_naive:mfcc11:mean',
         'audio_naive:mfcc12:mean',
         'audio_naive:mfcc0:std',
         'audio_naive:mfcc1:std',
         'audio_naive:mfcc2:std',
         'audio_naive:mfcc3:std',
         'audio_naive:mfcc4:std',
         'audio_naive:mfcc5:std',
         'audio_naive:mfcc6:std',
         'audio_naive:mfcc7:std',
         'audio_naive:mfcc8:std',
         'audio_naive:mfcc9:std',
         'audio_naive:mfcc10:std',
         'audio_naive:mfcc11:std',
         'audio_naive:mfcc12:std',
         'audio_properties:max_abs_value',
         'audio_properties:normalization_multiplier']

discrete = ['discrete:app_state:is_active',
            'discrete:app_state:is_inactive',
            'discrete:app_state:is_background',
            'discrete:app_state:missing',
            'discrete:battery_plugged:is_ac',
            'discrete:battery_plugged:is_usb',
            'discrete:battery_plugged:is_wireless',
            'discrete:battery_plugged:missing',
            'discrete:battery_state:is_unknown',
            'discrete:battery_state:is_unplugged',
            'discrete:battery_state:is_not_charging',
            'discrete:battery_state:is_discharging',
            'discrete:battery_state:is_charging',
            'discrete:battery_state:is_full',
            'discrete:battery_state:missing',
            'discrete:on_the_phone:is_False',
            'discrete:on_the_phone:is_True',
            'discrete:on_the_phone:missing',
            'discrete:ringer_mode:is_normal',
            'discrete:ringer_mode:is_silent_no_vibrate',
            'discrete:ringer_mode:is_silent_with_vibrate',
            'discrete:ringer_mode:missing',
            'discrete:wifi_status:is_not_reachable',
            'discrete:wifi_status:is_reachable_via_wifi',
            'discrete:wifi_status:is_reachable_via_wwan',
            'discrete:wifi_status:missing',
            'lf_measurements:light',
            'lf_measurements:pressure',
            'lf_measurements:proximity_cm',
            'lf_measurements:proximity',
            'lf_measurements:relative_humidity',
            'lf_measurements:battery_level',
            'lf_measurements:screen_brightness',
            'lf_measurements:temperature_ambient',
            'discrete:time_of_day:between0and6',
            'discrete:time_of_day:between3and9',
            'discrete:time_of_day:between6and12',
            'discrete:time_of_day:between9and15',
            'discrete:time_of_day:between12and18',
            'discrete:time_of_day:between15and21',
            'discrete:time_of_day:between18and24',
            'discrete:time_of_day:between21and3']

proc_magnet = ['proc_magnet:magnitude_spectrum:log_energy_band1',
                 'proc_magnet:3d:mean_x',
                 'proc_magnet:magnitude_spectrum:log_energy_band3',
                 'proc_magnet:3d:ro_yz',
                 'proc_magnet:magnitude_stats:percentile75',
                 'proc_magnet:magnitude_stats:value_entropy',
                 'proc_magnet:avr_cosine_similarity_lag_range0',
                 'proc_magnet:magnitude_autocorrelation:period',
                 'proc_magnet:3d:std_x',
                 'proc_magnet:magnitude_stats:moment3',
                 'proc_magnet:magnitude_stats:percentile25',
                 'proc_magnet:magnitude_spectrum:spectral_entropy',
                 'proc_magnet:magnitude_stats:std',
                 'proc_magnet:magnitude_stats:percentile50',
                 'proc_magnet:3d:ro_xy',
                 'proc_magnet:magnitude_stats:moment4',
                 'proc_magnet:avr_cosine_similarity_lag_range1',
                 'proc_magnet:avr_cosine_similarity_lag_range3',
                 'proc_magnet:magnitude_spectrum:log_energy_band4',
                 'proc_magnet:magnitude_spectrum:log_energy_band0',
                 'proc_magnet:3d:std_z',
                 'proc_magnet:magnitude_spectrum:log_energy_band2',
                 'proc_magnet:3d:mean_z',
                 'proc_magnet:3d:mean_y',
                 'proc_magnet:magnitude_autocorrelation:normalized_ac',
                 'proc_magnet:magnitude_stats:mean',
                 'proc_magnet:avr_cosine_similarity_lag_range4',
                 'proc_magnet:avr_cosine_similarity_lag_range2',
                 'proc_magnet:magnitude_stats:time_entropy',
                 'proc_magnet:3d:ro_xz',
                 'proc_magnet:3d:std_y'
                 ]

audio_rest = ['audio_naive:mfcc10:std',
                 'mfcc_7',
                 'audio_naive:mfcc1:std',
                 'audio_naive:mfcc3:mean',
                 'mfcc_8',
                 'audio_naive:mfcc6:mean',
                 'audio_naive:mfcc7:mean',
                 'mfcc_25',
                 'mfcc_23',
                 'mfcc_9',
                 'audio_naive:mfcc5:mean',
                 'mfcc_14',
                 'audio_naive:mfcc3:std',
                 'audio_naive:mfcc10:mean',
                 'mfcc_20',
                 'mfcc_15',
                 'audio_naive:mfcc9:std',
                 'audio_naive:mfcc7:std',
                 'audio_naive:mfcc6:std',
                 'mfcc_0',
                 'audio_naive:mfcc2:std',
                 'audio_naive:mfcc11:std',
                 'audio_naive:mfcc4:std',
                 'audio_naive:mfcc9:mean',
                 'audio_naive:mfcc2:mean',
                 'mfcc_5',
                 'mfcc_17',
                 'audio_naive:mfcc0:mean',
                 'mfcc_21',
                 'mfcc_1',
                 'audio_naive:mfcc5:std',
                 'mfcc_18',
                 'mfcc_4',
                 'audio_naive:mfcc4:mean',
                 'mfcc_13',
                 'audio_naive:mfcc1:mean',
                 'mfcc_22',
                 'mfcc_24',
                 'mfcc_12',
                 'audio_naive:mfcc0:std',
                 'mfcc_6',
                 'audio_naive:mfcc12:std',
                 'audio_naive:mfcc8:mean',
                 'audio_naive:mfcc12:mean',
                 'audio_naive:mfcc8:std',
                 'mfcc_2',
                 'audio_properties:max_abs_value',
                 'mfcc_19',
                 'mfcc_10',
                 'mfcc_3',
                 'mfcc_16',
                 'audio_naive:mfcc11:mean',
                 'mfcc_11',
                 'audio_properties:normalization_multiplier'
             ]

labels = ['Laying Down (action)',
          'Sneezing',
          'Talking On Phone',
          'Typing',
          'Phone in Pocket',
          'Standing up (action)',
          'Stairs - Going Up',
          'Jogging',
          'Running',
          'Sitting',
          'Sleeping',
          'Jumping',
          'Coughing',
          'Phone in Table, Facing Down',
          'Phone in Table, Facing Up',
          'Stairs - Going Down',
          'Phone in Hand',
          'Sitting Up (action)',
          'Phone in Bag',
          'Bathroom',
          'Walking',
          'Trembling',
          'Lying Down',
          'Standing',
          'Sitting Down (action)',
          'Assigned Prioception - Except Phone in Hand'
         ]


activity_label_names = [
          'Stairs - Going Up',
          'Jogging',
          'Running',
          'Sitting',
          'Sleeping',
          'Jumping',
          'Stairs - Going Down',
          'Walking',
          'Lying Down',
          'Standing'
         ]
transition_label_names = [
        'Laying Down (action)',
        'Sitting Down (action)',
        'Sitting Up (action)',
        'Standing up (action)'
        ]

def one_hot(y, labels):
    Y_onehot=[]
    for l in y:
        empty_label=np.zeros(len(labels))
        empty_label[labels.index(l)]=1.
        Y_onehot.append(empty_label)
    return(np.vstack(Y_onehot))


def get_metrics(target, output):
        
        pred = one_hot(np.argmax(output, axis=1), range(target.shape[1]))
        
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


class Custom_GRU(nn.Module):
    def __init__(self, config):
        super(Custom_GRU, self).__init__()
        self.input_size = config["input_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"] 
        self.output_size = config["output_dim"]
        self.learning_rate = config["learning_rate"]
        self.num_epochs = config["num_epochs"]
        self.num_directions = config["num_directions"]
        self.attention_setting = config["attention"]
        self.weight_decay = config["optimizer_decay"]
                
        if config["num_directions"] == 1:
            bidirectional = False
        elif config["num_directions"] == 2:
            bidirectional = True
        
        self.rnn = nn.GRU(input_size=self.input_size, num_layers= self.num_layers, hidden_size=self.hidden_size, 
                          batch_first=True, bidirectional=bidirectional, dropout=0.1).cuda()
        if(self.attention_setting == True):
            self.linear = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
            self.attention_weights = new_parameter(1, self.hidden_size*self.num_directions)
        else:
            self.linear = nn.Linear(self.hidden_size*self.num_directions, self.output_size)
        self.act = nn.Softmax()
        #self.act = nn.Tanh()
        
    def forward(self, x):
        pred, hidden = self.rnn(x, None)
        if(self.attention_setting == True):
            if(self.num_directions == 2):
                H = torch.cat((hidden[0], hidden[1]), 1)
            else:
                H = hidden
            M = F.tanh(H.squeeze(0))
            alpha = self.act(torch.mm( M, torch.transpose(self.attention_weights, 1, 0)))
            r = torch.mul(H , alpha.expand_as(H))
            h_wa = F.tanh(r)
            pred_out = self.act(self.linear(h_wa))
        else:
            pred_out = self.act(self.linear(pred)).unsqueeze(0).cuda()
        
        return pred_out

    def train_gru(self, train_inp, train_out, test_inp, test_act_out, test_out):
        predictions = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        #attention_optimizer = torch.optim.Adam([self.attention_weights], lr=self.learning_rate)
        
        #loss_func = nn.L1Loss()
        loss_func = F.binary_cross_entropy
        #attn_loss_fn = F.l1_loss
        train_loss = [] 
        for t in range(self.num_epochs):
            hidden = None
            inp = Variable(torch.from_numpy(train_inp.reshape((train_inp.shape[0], -1, self.input_size))).type(dtype), requires_grad=True)
            out = Variable(torch.from_numpy(train_out.reshape((train_inp.shape[0], self.output_size))).type(dtype))
        
            pred = self.forward(inp)
            optimizer.zero_grad()
            #attention_optimizer.zero_grad()
            predictions.append(pred.data.cpu().numpy())
            loss = loss_func(pred, out)
            #attn_loss = attn_loss_fn(self.attention_weights, torch.zeros_like(self.attention_weights))
            train_loss.append(loss.item())
            if t%100==0:
                print(t, loss.item())
            loss.backward()
            #attn_loss.backward()
            optimizer.step()
            #attention_optimizer.step()
        
        t_inp = Variable(torch.Tensor(test_inp.reshape((test_inp.shape[0], -1, self.input_size))).type(dtype), requires_grad=True)
        t_out = Variable(torch.Tensor(test_act_out.reshape((test_act_out.shape[0], self.output_size))).type(dtype))
        pred_t = self.forward(t_inp)
        test_loss = loss_func(pred_t, t_out)
        pred_numpy = pred_t.squeeze().data.cpu().numpy()
        
        filename= config["result_filename"].replace("Transition", "Activities")
        
        filename_modifier = 0
        while(os.path.exists("{}_predictions_probabilities_target_{}".format(filename, filename_modifier)+".csv")):
            filename_modifier+=1
        pred_fname="{}_predictions_probabilities_target_{}".format(filename, filename_modifier)+".csv"
        if(self.attention_setting == True):
            with open("{}_attention_weights_{}".format(filename, filename_modifier)+".csv", 'wb') as attn_file:
                np.savetxt(attn_file, self.attention_weights.data.cpu().numpy(), delimiter=",")
        
        with open(pred_fname, 'a') as f:
            #print(test_act_out.shape, pred_numpy.shape)
            predictions=np.vstack((list(activity_label_names)*3,np.hstack((test_act_out, one_hot(np.argmax(pred_numpy, axis=1), range(test_act_out.shape[1])), pred_numpy))))
            pd.DataFrame(predictions, columns=["Y"]*(len(activity_label_names))+["Y_Hat"]*(len(activity_label_names))+["Y_Hat_Prob"]*(len(activity_label_names))).to_csv(f, index=False)
        
        results_metrics = get_metrics(test_act_out, pred_numpy)
        results_conf_mat = confusion_matrix(np.argmax(test_act_out, axis=1), np.argmax(pred_numpy, axis=1))
        metric_names = np.array(["CV", "Balanced Accuracy", "Accuracy", "Error Rate", "Precision","Recall","Specificity", "F1", "TP","FP","FN","TN", "Micro F1","Macro F1","Weighted F1","Log-Loss","ROC AUC"])
        results = np.hstack((metric_names.reshape(-1, 1), np.vstack((activity_label_names, np.vstack(results_metrics)))))

        with open("{}.csv".format(filename), 'a') as f:
            pd.DataFrame(results).to_csv(f, header=False)
        with open("{}_conf_matrix.csv".format(filename), 'a') as f:  
            conf_mat_labels = activity_label_names[np.unique(np.argmax(test_act_out, axis=1))]
            #conf_mat_labels = activity_label_names
            pd.DataFrame(np.hstack((conf_mat_labels.reshape(-1, 1), results_conf_mat))).to_csv(f, header=False)
        
            #pd.DataFrame(np.hstack((activity_label_names.reshape(-1, 1), results_conf_mat))).to_csv(f, header=False)
        '''

        transition_act=np.where(np.argmax(pred_numpy, axis=1)==6, True, False)
        #print(transition_act)
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
            
        filename_modifier = 0
        while(os.path.exists("{}_predictions_probabilities_target_{}".format(filename, filename_modifier)+".csv")):
            filename_modifier+=1
        pred_fname="{}_predictions_probabilities_target_{}".format(filename, filename_modifier)+".csv"
        
        #print(test_out.shape, pred_one_hot.shape)
        with open(pred_fname, 'a') as f:
            predictions=np.vstack((list(transition_label_names)*2,np.hstack((test_out, pred_one_hot))))
            pd.DataFrame(predictions, columns=["Y"]*(len(transition_label_names))+["Y_Hat"]*(len(transition_label_names))).to_csv(f, index=False)
      
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
        plt.close()
        
        with open(filename+"_"+str(filename_modifier)+"_train_loss.csv", 'wb') as train_loss_file:
            wr = csv.writer(train_loss_file, quoting=csv.QUOTE_ALL)
            for tr_l in train_loss:
                wr.writerow([tr_l])
        '''
        final_loss = test_loss.detach().cpu().numpy().item()
        del inp
        del self.rnn
        del out
        del t_inp
        del t_out
        del test_loss
        del pred
        del pred_t
        del train_loss
        torch.cuda.empty_cache()
        plt.close()
        return final_loss

with open("/home/kchandrasekaran/wash/Datasets/WASH_User_Study/study_folds.pickle", 'rb') as handle:
    folds_dict = pickle.load(handle)
    
data_path = "/home/kchandrasekaran/wash/Datasets/WASH_User_Study/32_97_with_audio/"

    
hidden_layer_sizes=[512]
learning_rates=[1e-2, 1e-3]
epochs=[1000]
directions = [1, 2]
layers=[1]
attention = [True, False]
optimizer_decay = [0]
    
hyperparameters = [hidden_layer_sizes, directions, layers, epochs, learning_rates, attention, optimizer_decay]
all_parameter_combinations = list(itertools.product(*hyperparameters))
costs = []
for parameter_combo in all_parameter_combinations:
    config = {}
    config["input_dim"] = 201
    config["hidden_size"] = parameter_combo[0]
    config["num_directions"] = parameter_combo[1]
    config["num_layers"] = parameter_combo[2]
    config["output_dim"] = 11
    config["num_epochs"] = parameter_combo[3]
    config["learning_rate"] = parameter_combo[4]
    config["attention"] = parameter_combo[5]
    config["optimizer_decay"] = parameter_combo[6]
    if(config["attention"] == True):
        config["result_filename"] = "results/wash_attn/W_Decay_{}_Inverse_Transition_classification_w_attention_results_GRU_{}_directions_{}_layers_{}_lr_{}_units_{}_epochs".format(config["optimizer_decay"], config["num_directions"], config["num_layers"], config["learning_rate"], config["hidden_size"], config["num_epochs"])
    else:
        config["result_filename"] = "results/wash_attn/W_Decay_{}_Inverse_Transition_classification_results_GRU_{}_directions_{}_layers_{}_lr_{}_units_{}_epochs".format(config["optimizer_decay"], config["num_directions"], config["num_layers"], config["learning_rate"], config["hidden_size"], config["num_epochs"])    
    
    
   
    train_files = []
    test_files = []

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for i in range(5):
        
        train_test_split = folds_dict[i]
        train_files += train_test_split[0]
        test_files += train_test_split[1]

        activity_label_names = [
          'Stairs - Going Up',
          'Jogging',
          'Running',
          'Sitting',
          'Sleeping',
          'Jumping',
          'Stairs - Going Down',
          'Walking',
          'Lying Down',
          'Standing'
         ]
        transition_label_names = [
        'Laying Down (action)',
        'Sitting Down (action)',
        'Sitting Up (action)',
        'Standing up (action)'
        ]

        for filename in train_files:
            user = pd.read_csv(data_path + filename + '.csv', header=0, index_col= 0)
            train_data = train_data.append(user)

        for filename in test_files:
            user = pd.read_csv(data_path + filename + '.csv', header=0, index_col= 0)
            test_data = test_data.append(user)

        X_train = train_data[list(set(train_data.columns)^set(labels))]
        Y_train = train_data[labels]

        X_test = test_data[list(set(test_data.columns)^set(labels))]
        Y_test = test_data[labels]

        X_transition_train = np.nan_to_num(X_train.values)
        X_transition_validation = np.nan_to_num(X_test[:int(len(X_test)*0.5)].values)
        X_transition_test = np.nan_to_num(X_test[int(len(X_test)*0.5):].values)
        activities_labels_train = np.nan_to_num(train_data[activity_label_names].values)
        activities_labels_validation = np.nan_to_num(test_data[activity_label_names][:int(len(X_test)*0.5)].values)
        activities_labels_test = np.nan_to_num(test_data[activity_label_names][int(len(X_test)*0.5):].values)

        X_activities_train = np.nan_to_num(X_train.values)
        X_activities_validation = np.nan_to_num(X_test[:int(len(X_test)*0.5)].values)
        X_activities_test = np.nan_to_num(X_test[int(len(X_test)*0.5):].values)
        transition_labels_train = np.nan_to_num(train_data[transition_label_names].values)
        transition_labels_validation = np.nan_to_num(test_data[transition_label_names][:int(len(X_test)*0.5)].values)
        transition_labels_test = np.nan_to_num(test_data[transition_label_names][int(len(X_test)*0.5):].values)

        Y_activities_train = np.hstack((activities_labels_train, np.where(np.sum(activities_labels_train, axis=1) == 0, 1, 0).reshape(-1, 1)))
        Y_activities_validation = np.hstack((activities_labels_validation, np.where(np.sum(activities_labels_validation, axis=1) == 0, 1, 0).reshape(-1, 1)))
        Y_activities_test = np.hstack((activities_labels_test, np.where(np.sum(activities_labels_test, axis=1) == 0, 1, 0).reshape(-1, 1)))

        Y_transition_train = np.hstack((transition_labels_train, np.where(np.sum(transition_labels_train, axis=1) == 0, 1, 0).reshape(-1, 1)))
        Y_transition_validation = np.hstack((transition_labels_validation, np.where(np.sum(transition_labels_validation, axis=1) == 0, 1, 0).reshape(-1, 1)))
        Y_transition_test = np.hstack((transition_labels_test, np.where(np.sum(transition_labels_test, axis=1) == 0, 1, 0).reshape(-1, 1)))    

        transition_label_names = np.array(transition_label_names + ['Unknown'])
        activity_label_names = np.array(activity_label_names + ['Unknown'])

        #train_trans_inp, train_trans_out = X_transition_train, Y_transition_train
        #val_trans_inp, val_trans_out = X_transition_validation, Y_transition_validation
        #test_trans_inp, test_trans_out = X_transition_test, Y_transition_test

        #train_act_inp, train_act_out = X_activities_train, Y_activities_train
        #val_act_inp, val_act_out = X_activities_validation, Y_activities_validation
        #test_act_inp, test_act_out = X_activities_test, Y_activities_test

        #try:
        start_time = time.time()
        model = Custom_GRU(config)
        model = nn.DataParallel(model).to(device)
        loss = model.train_gru( X_activities_train, Y_activities_train, X_activities_validation, Y_activities_validation, Y_transition_validation)
        costs.append(loss)
        config["CV_index"] = i
        config["test_loss"] = loss
        config["time_elapsed"] = time.time()-start_time
        with open("results/wash_attn/Wash_tune_results.csv", 'a') as f:    
            #print(config)
            pd.DataFrame(config, index=[0]).to_csv(f, header=False)
        del model
        torch.cuda.empty_cache()
        #except:
        #    pass
