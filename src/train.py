import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from model import Dip_l, Dip_c, Dip_g

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Dip-l", required=True, choices=["Dip_l", "Dip_g", "Dip_c"],
                    help="model")
parser.add_argument("--hidden_dim", type=int, default=128, help="hidden_dim")
parser.add_argument('--bi_direction', action="store_true", default=True, help="bi_direction")
parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
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
    eval_model.eval()
    y_label = []
    y_pred = []

    with torch.no_grad():
        for visit, dignose in dataloader:
            visit, dignose = visit.to(device), dignose.to(device)
            outputs = eval_model(visit)
            pred = outputs.data > 0.5
            y_label.append(dignose.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    # concat
    y_label = np.concatenate(y_label, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # calc: precision, recall, f1
    micro_precision = precision_score(y_label, y_pred, average="micro")
    macro_precision = precision_score(y_label, y_pred, average="macro")

    micro_recall = recall_score(y_label, y_pred, average="micro")
    macro_recall = recall_score(y_label, y_pred, average="macro")

    micro_f1 = f1_score(y_label, y_pred, average='micro')
    macro_f1 = f1_score(y_label, y_pred, average='macro')

    return micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1


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

    # Load Dataset:
    features = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))

    train_data = TensorDataset(features, labels)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    eval_data = TensorDataset(features, labels)
    eval_loader = DataLoader(eval_data, batch_size=32, shuffle=True)

    test_data = TensorDataset(features, labels)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    input_dim = 10  # feature
    output_dim = 2  # 输出的维度，假设是2分类
    if args.model == "Dip_l":
        model = Dip_l(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      bi_direction=args.bi_direction) # 默认为True
    elif args.model == "Dip_g":
        model = Dip_g(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      bi_direction=args.bi_direction) # 默认为True
    else:  # model: "Dip_c"
        model = Dip_c(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      max_timesteps=10, # 这里不知道max_timesteps具体的作用
                      bi_direction=args.bi_direction) # 默认为True

    epoch = 10
    # loss_fn = nn.CrossEntropyLoss()
    optimzer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for i in range(epoch):
        total_loss = 0
        for x, y in train_loader:
            optimzer.zero_grad()
            output = model(x)
            # loss = loss_fn
            # 这里还需要补充邻接矩阵的信息，之前只使用CrossEntropyLoss
            loss = kg_loss(output, y, model, A=[1], beta=args.beta, batch_size=args.batch_size)
            loss.backward()
            optimzer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {i}, Average Loss: {avg_loss}")

        # eval_model:
        micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1 = evaluate(model, eval_loader, device)
        print(f"micro_p: {micro_precision}, macro_p:{macro_precision}"
              f"micro_r: {micro_recall}, macro_r:{macro_recall}"
              f"micro_fi: {micro_f1}, macro_f1:{macro_f1}")

        # save_model:
        model_name = f"Epoch_{i}.model"
        with open(os.path.join(saved_path, model_name), "wb") as model_file:
            torch.save(model.state_dict(), model_file)

        # test:
        micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1 = evaluate(model, test_loader, device)
        print(f"micro_p: {micro_precision}, macro_p:{macro_precision}"
              f"micro_r: {micro_recall}, macro_r:{macro_recall}"
              f"micro_fi: {micro_f1}, macro_f1:{macro_f1}")