import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from Dataset import DiseasePredDataset, read_data, load_dataset
from Gram import GRAM
from HAP import HAP
from utils import llprint, get_accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument("--input_dim", type=int, default=256, help="input_dim (feature_size)")
parser.add_argument("--embed_dim", type=int, default=1024, help="embedding_dim")
parser.add_argument("--hidden_dim", type=int, default=256, help="hidden_dim")
parser.add_argument("--attn_dim", type=int, default=128, help="attention_dim")
parser.add_argument("--output_dim", type=int, default=90, help="output_dim (disease_size)")
parser.add_argument('--batch_size', type=int, default=8, help="batch_size")

args = parser.parse_args()


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
            x, p2c, c2p, indicator1, indicator2, y = batch
            
            x = x.to(device)
            p2c = p2c.to(device)
            c2p = c2p.to(device)
            indicator1 = indicator1.to(device)
            indicator2 = indicator2.to(device)
            y = y.to(device)

            outputs = eval_model(x, p2c, c2p, indicator1, indicator2)
            loss = loss_fn(outputs, y)
            y_label.extend(np.array(y.data.cpu()))  # 放到正确的label中
            y_pred.extend(np.array(outputs.data.cpu()))
            # llprint('\rtest step: {} / {}'.format(idx, len(dataloader)))
    total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = get_accuracy(y_label, y_pred)
    
    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, p2c, c2p, indicator1s, indicator2s, labels = read_data("../../data/ancestor_information.pkl",
                                                              "../../data/label_one_hot.pt")
    

    split_train_point = int(len(features) * 6.7 / 10)
    split_test_point = int(len(features) * 8.7 / 10)

    train_dataset, valid_dataset, test_dataset = load_dataset(features, p2c, c2p, indicator1s, indicator2s, labels, split_train_point, split_test_point) 

    train_loader = DataLoader(train_dataset)
    valid_loader = DataLoader(valid_dataset)
    test_loader = DataLoader(test_dataset)

    model = HAP(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_ancestors=72,
        output_dim=args.output_dim,
        attn_dim=args.attn_dim
    )

    epoch=50
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
            x, p2c, c2p, indicator1, indicator2, y = batch
            
            x = x.to(device)
            p2c = p2c.to(device)
            c2p = c2p.to(device)
            indicator1 = indicator1.to(device)
            indicator2 = indicator2.to(device)
            y = y.to(device)

            optimzer.zero_grad()
            output = model(x, p2c, c2p, indicator1, indicator2)
            loss = loss_fn(output, y)
            
            loss.backward()
            optimzer.step()
            llprint('\rtraining step: {} / {}'.format(idx, len(train_loader)))
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