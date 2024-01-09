import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target
import pickle
from tqdm import tqdm
import torch
from Dataset import read_data


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def get_accuracy(y, pred):
    y = np.array(y)
    pred = np.array(pred)

    y_label = np.zeros(y.shape)
    y_pred = np.zeros(pred.shape)
    y_label[y >= 0.5] = 1
    y_label[y < 0.5] = 0
    y_pred[pred >= 0.5] = 1
    y_pred[pred < 0.5] = 0

    print(y_label.sum(axis=0))
    print(y_label.sum(axis=0).shape)
    # input()

    # 删掉那个没有疾病的维度
    
    y_label = np.delete(y_label, obj=32, axis=1)
    y_pred = np.delete(y_pred, obj=32, axis=1)
    
    # Micro Acc & Macro Acc
    micro_accuracy = roc_auc_score(y_label, y_pred, average="micro")
    macro_accuracy = roc_auc_score(y_label, y_pred, average="macro")

    precision_mean = precision_score(y_label, y_pred, average="micro")
    recall_mean = recall_score(y_label, y_pred, average="micro")
    f1_mean = f1_score(y_label, y_pred, average="micro")
    return micro_accuracy, macro_accuracy, precision_mean, recall_mean, f1_mean

def load_W_index(data):
    W_index_list = []
    for sample in data:
        W_index_list.append(sample['GAT_data']['W_index_list'])
    return W_index_list
