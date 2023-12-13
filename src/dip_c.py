import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from Dataset import DiseasePredDataset, read_data
from model import Dip_l, Dip_c, Dip_g, Retain
from utils import llprint, get_accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Dip_l", choices=["Dip_l", "Dip_g", "Dip_c"],
                    help="model")
parser.add_argument("--input_dim", type=int, default=2850, help="input_dim (feature_size)")
parser.add_argument("--hidden_dim", type=int, default=128, help="hidden_dim")
parser.add_argument("--output_dim", type=int, default=90, help="output_dim (disease_size)")
parser.add_argument('--bi_direction', action="store_true", default=True, help="bi_direction")
parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
parser.add_argument('--beta', type=float, default=0.5, help="KG factor in loss")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

args = parser.parse_args()

saved_path = "../saved_model/"
if not os.path.exists(saved_path):
    os.makedirs(saved_path)


def evaluate(eval_model, dataloader, device):
    """
    在一次batch中
    y_pred: (batch_size, dignosis_size)
    an exampe: [[1,0,0,0,0,1,...], [0,1,0,0,1,0,1], ...]
    """
    loss_fn = nn.CrossEntropyLoss()
    eval_model.eval()
    y_label = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            visit, dignose = batch
            visit, dignose = visit.to(device), dignose.to(device)
            outputs = eval_model(visit)
            loss = loss_fn(outputs, dignose)
            y_label.extend(np.array(dignose.data.cpu()))  # 放到正确的label中
            y_pred.extend(np.array(outputs.data.cpu()))
            # llprint('\rtest step: {} / {}'.format(idx, len(dataloader)))
    total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = get_accuracy(y_label, y_pred)
    def calc_marco(label, pred):
        """
        之前的直接用sklearn里面的average=“macro”是不对的
        现在针对下面这种 batch_size * diagnoses_size
        应该先转置，针对每个diagnoses_size计算出acc P R F1, 在直接取平均
        # y_test = [
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        # ]
        # y_pred = [
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        #     [1, 0, 0, 0, 1, 1, 1, 0],
        # ]
        """
        y_marco_label = np.array(label).T
        y_marco_pred = np.array(pred).T

        macro_acc = []
        macro_precision = []
        macro_recall = []
        macro_f1 = []

        for i in range(len(y_marco_pred)):
            y_pred_i = y_marco_pred[i]  # 第i个疾病的预测
            y_label_i = y_marco_label[i]  # 第i个疾病的标签

            macro_acc.append(accuracy_score(y_label_i, y_pred_i))
            macro_precision.append(precision_score(y_label_i, y_pred_i, average="micro"))
            macro_recall.append(recall_score(y_label_i, y_pred_i, average="micro"))
            macro_f1.append(f1_score(y_label_i, y_pred_i, average="micro"))

        macro_acc = np.mean(macro_acc)
        macro_precision = np.mean(macro_precision)
        macro_recall = np.mean(macro_recall)
        macro_f1 = np.mean(macro_f1)
        return macro_acc, macro_precision, macro_recall, macro_f1

    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean


def kg_loss(output, target, model: Dip_l, A, beta, batch_size):
    """
    增添的知识图谱关系的loss
    A: 关系邻接矩阵
    """
    hidden_size = model.hidden_dim
    # 先计算原本的交叉熵loss
    ce_loss = nn.CrossEntropyLoss()
    loss1 = ce_loss(output, target)

    # RNN层的权重W
    W = model.rnn.weight_ih_l0  # (3*hidden_size, input_size)
    W_ir = W[0:hidden_size, :]  # (hidden_size, input_size)
    W_iz = W[hidden_size:2 * hidden_size, :]
    W_in = W[2 * hidden_size:3 * hidden_size, :]
    W_matrix = [W_ir, W_iz, W_in]

    relationship_loss = 0
    # A: (input_size , input_size) 01matrix
    for i in range(W_ir.shape[0]):
        for j in range(i, W_ir.shape[0]):
            if A[i, j] == 1:
                for weight in W_matrix:
                    relationship_loss += torch.norm(weight[i] - weight[j], 2) ** 2
    total_loss = loss1 + (beta / batch_size) * relationship_loss  # 这里要除以batch_size
    return total_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = read_data(feature_file="../data/features_one_hot.pt",
                                 label_file='../data/label_one_hot.pt')
    

    split_train_point = int(len(features) * 6.7 / 10)
    split_test_point = int(len(features) * 8.7 / 10)
    train_features, train_labels = features[:split_train_point], labels[:split_train_point]
    test_features, test_labels = features[split_train_point:split_test_point], labels[split_train_point:split_test_point]
    valid_features, valid_labels = features[split_test_point:], labels[split_test_point:]

    train_data = DiseasePredDataset(train_features, train_labels)
    test_data = DiseasePredDataset(test_features, test_labels)
    valid_data = DiseasePredDataset(valid_features, valid_labels)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    input_dim = args.input_dim  # feature
    output_dim = args.output_dim
    if args.model == "Dip_l":
        model = Dip_l(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      bi_direction=args.bi_direction)  # 默认为True
    elif args.model == "Dip_g":
        model = Dip_g(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      bi_direction=args.bi_direction)  # 默认为True
    else:  # model: "Dip_c"
        model = Dip_c(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      max_timesteps=10,  # 这里不知道max_timesteps具体的作用
                      bi_direction=args.bi_direction)  # 默认为True

    epoch = 100
    loss_fn = nn.CrossEntropyLoss()
    optimzer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)

    best_eval_macro_auc = 0
    best_eval_epoch = 0

    best_test_macro_auc = 0
    best_test_epoch = 0
    for i in range(epoch):
        print('\nepoch {} --------------------------'.format(i))
        total_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimzer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            # 这里还需要补充邻接矩阵的信息，之前只使用CrossEntropyLoss
            # loss = kg_loss(output, y, model, A=[1], beta=args.beta, batch_size=args.batch_size)
            loss.backward()
            optimzer.step()
            # llprint('\rtraining step: {} / {}'.format(idx, len(train_loader)))
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {i}, Average Loss: {avg_loss}")

        # eval:
        macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = \
            evaluate(model, valid_loader, device)
        print(f"\nValid Result:\n"
              f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}"
              f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}")
        if macro_auc > best_eval_macro_auc:
            best_eval_macro_auc = macro_auc
            best_eval_epoch = i
        
        # test:
        macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = \
            evaluate(model, test_loader, device)
        print(f"\nTest Result:\n"
              f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}"
              f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}")
        
        if macro_auc > best_test_macro_auc:
            best_test_macro_auc = macro_auc
            best_test_epoch = i
        # save_model:
        model_name = f"Epoch_{i}.model"
        with open(os.path.join(saved_path, model_name), "wb") as model_file:
            print()
            # torch.save(model.state_dict(), model_file)
    print(f"Best Eval Epoch:{best_eval_epoch}, best_Macro_auc:{best_eval_macro_auc}")
    print(f"Best Test Epoch:{best_test_epoch}, best_Macro_auc:{best_test_macro_auc}")




# y_test = [0, 0, 1, 0, 1]
# y_predict = [0, 1, 1, 0, 0]
#
# print('准确率:', accuracy_score(y_test, y_predict))  # 预测准确率输出
#
# print('宏平均精确率:', precision_score(y_test, y_predict, average='macro'))  # 预测宏平均精确率输出
# print('微平均精确率:', precision_score(y_test, y_predict, average='micro'))  # 预测微平均精确率输出
#
# print('宏平均召回率:', recall_score(y_test, y_predict, average='macro'))  # 预测宏平均召回率输出
# print('微平均召回率:', recall_score(y_test, y_predict, average='micro'))  # 预测微平均召回率输出
#
# print('宏平均F1-score:', f1_score(y_test, y_predict, labels=[0, 1], average='macro'))  # 预测宏平均f1-score输出
# print('微平均F1-score:', f1_score(y_test, y_predict, labels=[0, 1], average='micro'))  # 预测微平均f1-score输出

# calc macro average:
# y_test = [
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
# ]
# y_pred = [
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 1, 1, 1, 0],
# ]
# y_test = np.array(y_test)
# y_pred = np.array(y_pred)
#
# y_marco_label = y_test.T
# y_marco_pred = y_pred.T
#
# print(y_marco_label)
# print(y_marco_pred)
#
# macro_acc = []
# macro_precision = []
# macro_recall = []
# macro_f1 = []
#
# for i in range(len(y_marco_pred)):
#     y_pred_i = y_marco_pred[i]  # 第i个疾病的预测
#     y_label_i = y_marco_label[i]  # 第i个疾病的标签
#
#     macro_acc.append(accuracy_score(y_label_i, y_pred_i))
#     macro_precision.append(precision_score(y_label_i, y_pred_i, average="micro"))
#     macro_recall.append(recall_score(y_label_i, y_pred_i, average="micro"))
#     macro_f1.append(f1_score(y_label_i, y_pred_i, average="micro"))
#
# macro_acc = np.mean(macro_acc)
# macro_precision = np.mean(macro_precision)
# macro_recall = np.mean(macro_recall)
# macro_f1 = np.mean(macro_f1)

main()