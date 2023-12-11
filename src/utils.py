import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def get_accuracy(y, pred):
    acc_list = []
    y = np.array(y)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    pred = np.array(pred)
    pred_res = np.zeros(pred.shape)
    pred_res[pred >= 0.5] = 1
    pred_res[pred < 0.5] = 0

    print(y.shape)
    print(pred_res.shape)
    print(y)
    print(pred_res)
    print(type_of_target(y))
    print(type_of_target(pred_res))
    input()
    macro_auc = accuracy_score(y, pred_res)
    micro_auc = accuracy_score(y, pred_res)
    # micro_auc = roc_auc_score(y, pred_res, average="micro")
    # average_auc = roc_auc_score(y, pred_res)

    precision_mean, recall_mean, f1_mean = get_precision_recall_f1(y, pred_res)
    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean


def get_precision_recall_f1(y, pred):
    precision_list, recall_list, f1_list = [], [], []
    if y.ndim == 1:
        precision_list.append(precision_score(y, pred))
        recall_list.append(recall_score(y, pred))
        f1_list.append(f1_score(y, pred))
    else:
        for col in range(y.shape[1]):
            precision_list.append(precision_score(y[:, col], pred[:, col], average="micro"))
            recall_list.append(recall_score(y[:, col], pred[:, col], average="micro"))
            f1_list.append(f1_score(y[:, col], pred[:, col], average="micro"))
    return np.mean(np.array(precision_list)), np.mean(np.array(recall_list)), np.mean(np.array(f1_list))



