import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from Dataset import DiseasePredDataset, read_data
from models import Dip_l, Dip_c, Dip_g, Retain, LSTM_Model
from utils import llprint, get_accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Dip_c", choices=["Dip_l", "Dip_g", "Dip_c", "Retain", "LSTM"],
                    help="model")
parser.add_argument("--input_dim", type=int, default=2850, help="input_dim (feature_size)")
parser.add_argument("--hidden_dim", type=int, default=128, help="hidden_dim")
parser.add_argument("--output_dim", type=int, default=90, help="output_dim (disease_size)")
parser.add_argument('--bi_direction', action="store_true", default=True, help="bi_direction")
parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
parser.add_argument('--beta', type=float, default=0.5, help="KG factor in loss")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

args = parser.parse_args()

saved_path = "../../saved_model/baseline/"
model_name = args.model
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
            outputs = eval_model(visit)
            loss = loss_fn(outputs, dignose)
            y_label.extend(np.array(dignose.data.cpu()))  # 放到正确的label中
            y_pred.extend(np.array(outputs.data.cpu()))
            # llprint('\rtest step: {} / {}'.format(idx, len(dataloader)))
    total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = get_accuracy(y_label, y_pred)
    
    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = read_data(feature_file="../../data/features_one_hot.pt",
                                 label_file='../../data/label_one_hot.pt')
    

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
    elif args.model == "Dip_c":  # model: "Dip_c"
        model = Dip_c(input_dim=input_dim,
                      hidden_dim=args.hidden_dim,
                      output_dim=output_dim,
                      max_timesteps=10,  # 这里不知道max_timesteps具体的作用
                      bi_direction=args.bi_direction)  # 默认为True
    elif args.model == "Retain":  # model: Retain
        model = Retain(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            device=device
        )
    else:
        model = LSTM_Model(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim
        )

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
        epoch_name = f"Epoch_{i}.model"
        # with open(os.path.join(saved_path, epoch_name), "wb") as model_file:
        #     print()
        #     # torch.save(model.state_dict(), model_file)
    print(f"Best Eval Epoch:{best_eval_epoch}, best_Macro_auc:{best_eval_macro_auc}")
    print(f"Best Test Epoch:{best_test_epoch}, best_Macro_auc:{best_test_macro_auc}")

main()