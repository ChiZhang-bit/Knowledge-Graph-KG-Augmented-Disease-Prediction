import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from Dataset import DiseasePredDataset, read_data
from model.DiseasePredModel import DiseasePredModel
from utils import llprint, get_accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Dip_g", choices=["Dip_l", "Dip_g", "Dip_c", "Retain", "LSTM"],
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
model_name = "Our_model"
path = os.path.join(saved_path, model_name)
if not os.path.exists(path):
    os.makedirs(path)


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
            lstm_out, kg_out, output = eval_model(visit)
            loss = loss_fn(output, dignose)
            y_label.extend(np.array(dignose.data.cpu()))  # 放到正确的label中
            y_pred.extend(np.array(output.data.cpu()))
            # llprint('\rtest step: {} / {}'.format(idx, len(dataloader)))
    total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = get_accuracy(y_label, y_pred)

    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean


def regularization_loss(output, target, model, adj, beta, batch_size):
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
            if adj[i, j] != 0:
                for weight in W_matrix:
                    relationship_loss += torch.norm(weight[i] - weight[j], 2) ** 2
    total_loss = loss1 + (beta / batch_size) * relationship_loss  # 这里要除以batch_size
    return total_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = read_data(feature_file="../data/features_one_hot.pt",
                                 label_file='../data/label_one_hot.pt')
    with open('../data/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)

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
    hidden_dim = args.hidden_dim
    model = DiseasePredModel(
        dipole_type=args.model,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        bi_direction=args.bi_direction
    )

    epoch = 100
    loss_kg = nn.CrossEntropyLoss()
    optimzer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
    adj = torch.zeros(size = (2850, 2850))

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
            lstm_out, kg_out, output = model(x, adj)
            
            lstm_loss = regularization_loss(lstm_out, y, adj, beta=args.beta, batch_size=args.batch_size)
            pkgat_loss = loss_kg(kg_out, y)

            # 这里还需要补充邻接矩阵的信息，之前只使用CrossEntropyLoss
            # loss = kg_loss(output, y, model, adj, beta=args.beta, batch_size=args.batch_size)
            # loss = loss_fn(output, y)
            loss = lstm_loss + pkgat_loss
            loss.backward()
            optimzer.step()
            llprint('\rtraining step: {} / {}'.format(idx, len(train_loader)))
            input()
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
        epoch_name = f"Epoch_{i}.model"
        # with open(os.path.join(saved_path, epoch_name), "wb") as model_file:
        #     print()
        #     # torch.save(model.state_dict(), model_file)
    print(f"Best Eval Epoch:{best_eval_epoch}, best_Macro_auc:{best_eval_macro_auc}")
    print(f"Best Test Epoch:{best_test_epoch}, best_Macro_auc:{best_test_macro_auc}")


main()
