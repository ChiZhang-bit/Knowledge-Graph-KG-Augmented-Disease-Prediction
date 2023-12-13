import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target


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

    # print(y_label.sum(axis=0))
    # print(y_label.sum(axis=0).shape)
    # 删掉那个没有疾病的维度
    y_label = np.delete(y_label, obj=32, axis=1)
    y_pred = np.delete(y_pred, obj=32, axis=1)
    # print(y_label.sum(axis=0))
    # print(y_label.sum(axis=0).shape)
    # input()
    
    # Micro Acc & Macro Acc
    micro_accuracy = roc_auc_score(y_label, y_pred, average="micro")
    macro_accuracy = roc_auc_score(y_label, y_pred, average="macro")

    precision_mean = precision_score(y_label, y_pred, average="micro")
    recall_mean = recall_score(y_label, y_pred, average="micro")
    f1_mean = f1_score(y_label, y_pred, average="micro")
    return micro_accuracy, macro_accuracy, precision_mean, recall_mean, f1_mean

def get_marco_micro_acc(y:np.array, pred:np.array):
    """
    y: [samples, 100]
    pred: [samples, 100]
    """
    micro_acc_list = []
    for i in range(y.shape[0]):
        micro_acc_list.append(accuracy_score(y[i], pred[i]))
    micro_acc = np.mean(micro_acc_list)

    trans_y = y.T
    trans_pred = pred.T
    macro_acc_list = []
    for i in range(trans_y.shape[0]):
        macro_acc_list.append(accuracy_score(trans_y[i], trans_pred[i]))
    macro_acc = np.mean(macro_acc_list)
    # print(micro_acc)
    # print(macro_acc)
    return micro_acc, macro_acc

