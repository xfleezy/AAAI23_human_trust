import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from baselines import *
import numpy as np

finetune_trust = None

def train_rnn(train_loader, test_loader, epoch):
    input_size, trust_level = train_loader[0][0].shape[0], train_loader[0][1].shape[0]
    model = RNN(input_size=input_size, trust_level=trust_level)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    best_acc = 0
    best_auc = 0
    best_f1 = 0
    for e in range(epoch):
        model.train()
        for i, (input, gr_de, gr_trust, mask) in enumerate(train_loader):
            input = torch.permute(input, (1, 0, 2)).float()
            decision, trust = model(input.cuda())
            if finetune_trust:
                loss1 = ce(decision, gr_de.cuda())
                loss2 = torch.mean(-torch.log(torch.sum(trust * gr_trust.cuda(), dim=-1, keepdim=True)) * mask.cuda())
                loss = loss1 + loss2
            else:
                loss = ce(decision, gr_de.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        label = np.array([0])
        pred = np.array([0])
        prob = np.array([0])
        for i, (input, gr_de, gr_trust, mask) in enumerate(test_loader):
            input = torch.permute(input, (1, 0, 2)).float()
            decision, trust = model(input.cuda())
            prob = np.concatenate([prob, decision.float().cpu().detach().numpy()])
            pred = np.concatenate([pred, np.argmax(prob, axis=1)])
            label = np.concatenate([label, gr_de.float().numpy()])

        best_acc = max(best_acc, accuracy_score(label, pred))
        best_f1 = max(best_f1, f1_score(label, pred))
        best_auc = max(best_auc, roc_auc_score(label, prob))

    return best_acc, best_f1, best_auc


def train_mlp(train_loader, test_loader, epoch):
    input_size, trust_level = train_loader[0][0].shape[0], train_loader[0][1].shape[0]
    model = MLP(input_size=input_size)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    best_acc = 0
    best_auc = 0
    best_f1 = 0
    for e in range(epoch):
        model.train()
        for i, (input, gr_de, gr_trust, mask) in enumerate(train_loader):
            input = torch.permute(input, (1, 0, 2)).float()
            decision = model(input.cuda())
            loss = ce(decision, gr_de.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        for i, (input, gr_de, gr_trust, mask) in enumerate(test_loader):
            input = torch.permute(input, (1, 0, 2)).float()
            decision = model(input.cuda())
            prob = decision.float().cpu().detach().numpy()
            pred = np.argmax(prob, axis=1)
            label = gr_de.float().numpy()

        best_acc = max(best_acc, accuracy_score(label, pred))
        best_f1 = max(best_f1, f1_score(label, pred))
        best_auc = max(best_auc, roc_auc_score(label, prob))
    return best_acc, best_f1, best_auc