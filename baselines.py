import sklearn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
class human_dataset(Dataset):

    def __init__(self, X,Y):
        self.x_train = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y_train = [torch.tensor(y, dtype=torch.float32) for y in Y]

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class MLP(nn.Module):

    def __init__(self, input_size, trust_level=5):
        super(MLP, self).__init__()
        self.in_layer = nn.Linear(input_size, 16)
        self.trust_layer = nn.Linear(16, trust_level)
        self.decision_layer = nn.Linear(16, 2)

    def forward(self, x, finetune_trust=False):
        x = F.relu(self.in_layer(x))
        d = self.decision_layer(x)
        if finetune_trust:
            trust = self.trust_layer(x)
            return F.softmax(d, dim=-1), F.softmax(trust, dim=-1)
        return F.softmax(d, dim=-1)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=16, trust_level=5):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=False)
        # self.trust_layer = nn.Linear(hidden_size, trust_level)
        self.decision_layer = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output, hidden = self.rnn(x, None)
        # output = torch.permute(output, (1, 0, 2))
        # trust = F.softmax(self.trust_layer(output), dim=-1)
        d = self.decision_layer(output)
        decision = F.softmax(self.decision_layer(output),dim=-1)
        # trust_loss = -torch.log(torch.sum(trust * trust_label, dim=-1, keepdim=True)) * mask
        return decision


def logistic(data, label, iteration=10):
    lr = LogisticRegression()
    avg_acc = []
    avg_auc = []
    avg_f1 = []
    for i in range(iteration):
        train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=2022 + i)
        lr.fit(train_x, train_y)
        pred = lr.predict(test_x)
        prob = lr.predict_proba(test_x)
        acc = accuracy_score(test_y, pred)
        auc = roc_auc_score(test_y, prob)
        f1 = f1_score(test_y, pred, average='macro')
        avg_acc.append(acc)
        avg_auc.append(auc)
        avg_f1.append(f1)

    metrics = {"acc": [np.mean(avg_acc), np.std(avg_acc)], "auc": [np.mean(avg_auc), np.std(avg_auc)],
               "f1": [np.mean(avg_f1), np.std(avg_f1)]}
    return metrics

def train_model(model, train_loader, optimizer):
    model.train()
    all_loss = 0
    ce = nn.CrossEntropyLoss()
    for i, (input, gr) in enumerate(train_loader):
        input = input.float()
        output = model(input.cuda())
        o = output.reshape(-1,2)
        g = gr.cuda().reshape(-1)
        g = g.type(torch.LongTensor)
        loss = ce(o.cuda(),g.cuda())
        all_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print("epoch:{}  avg_Loss:{}".format(epoch, all_loss / (i + 1)))

def test_model(model, test_loader):
    error = 0
    model.eval()
    F1 = []
    Confusion = []
    FPR = []
    FNR = []
    import numpy as np
    for i, (input, gr) in enumerate(test_loader):
        output = model(input.cuda().float()).float().cpu().detach().numpy().reshape(20,2)

        pred = np.argmax(output, axis=1).tolist()

        # output = output.float().cpu().detach().numpy()
        ground_truth = gr.float().numpy().reshape(20).tolist()
        # print(output.shape)
        # ground_truth = np.argmax(ground_truth, axis=1)
        f1 = f1_score(pred, ground_truth, average="macro")
        F1.append(f1)
        if sum(ground_truth) == 20:
            ground_truth.append(0)
            pred.append(0)
        c = confusion_matrix(ground_truth, pred)
        c = c.astype('float') / c.sum(axis=1)[:, np.newaxis]
        FPR.append(c[0, 1])
        FNR.append(c[1, 0])
    # print("avg acc:{}".format(acc/(i+1)))
    print("method:{}, F1_w:{}".format("rnn", sum(F1)/len(F1)))
    print("method:{}, FPR:{}".format("rnn", sum(FPR)/len(FPR)))
    print("method:{}, FNR:{}".format("rnn", sum(FNR)/len(FNR)))


if __name__ == '__main__':
    import re
    import copy
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn import preprocessing
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd


    class human(object):
        def __init__(self, data=0, demo=0, ini_trust=0, mid_trust=0, final_trust=None):
            self.data = data
            self.demo = demo
            self.ini_trust = ini_trust
            self.mid_trust = mid_trust
            self.final_trust = final_trust


    def string_process(string):
        res = re.sub(r'[^\w\s]', '', string)
        return res.lower()


    def normalize(raw_data):
        data = copy.copy(raw_data)
        min_max_scalar = preprocessing.MinMaxScaler()
        for elem in data.columns:
            elem_values = data[elem].values
            temp_scaled = min_max_scalar.fit_transform(elem_values.reshape((len(elem_values), 1)))
            data[elem] = temp_scaled
        return data


    def one_hot(onehot_data):
        res = []
        # le encoder and enc encoder
        enc = OneHotEncoder(handle_unknown='ignore')
        le = preprocessing.LabelEncoder()
        feature_label = [list(set(elem)) for elem in onehot_data.values.transpose()]
        for i in range(len(onehot_data.columns)):
            # fit and transform
            le.fit(feature_label[i])
            le_transform = le.transform(onehot_data[onehot_data.columns[i]])
            set_le_transform = np.array(list(set(le_transform)))
            enc.fit(set_le_transform.reshape((len(set_le_transform), 1)))
            res.append(enc.transform(le_transform.reshape((len(le_transform), 1))))
        one_hot_df_list = [pd.DataFrame(elem.toarray()) for elem in res]
        onehot_feature = onehot_data.columns
        for i in range(len(onehot_feature)):
            for cag in one_hot_df_list[i].columns:
                one_hot_df_list[i] = one_hot_df_list[i].rename(columns={cag: onehot_feature[i] + str(cag)})
        one_hot_df = pd.concat(one_hot_df_list, axis=1)
        return one_hot_df


    ini_s = pd.read_json("initialSurvey_0503_01.json", lines=True)
    mid_s = pd.read_json("misSurvey_0503_01.json", lines=True)
    fin_s = pd.read_json("surveys_0503_01.json", lines=True)
    data = pd.read_json("pilotStudy_0503_01.json", lines=True)
    data_pool = pd.read_csv("sample_data_0329.csv")
    # data = data.query("treatment==1")

    for i in data_pool:
        data_pool.loc[(data_pool['age'] > 16) & (data_pool['age'] <= 25), 'age'] = 1
        data_pool.loc[(data_pool['age'] > 25) & (data_pool['age'] <= 32), 'age'] = 2
        data_pool.loc[(data_pool['age'] > 32) & (data_pool['age'] <= 40), 'age'] = 3
        data_pool.loc[(data_pool['age'] > 40) & (data_pool['age'] <= 50), 'age'] = 4
        data_pool.loc[data_pool['age'] > 50, 'age'] = 5

    label = data_pool[["income", "ml_pred", "id"]]
    data_pool = data_pool.drop(["income", "ml_pred", "id"], axis=1)
    numerical_var = [c for c in data_pool.columns if data_pool[c].dtype != object]
    string_var = [c for c in data_pool.columns if data_pool[c].dtype == object]
    data_pool = data_pool[data_pool.columns]
    numerical_data = data_pool[numerical_var]
    string_data = data_pool[string_var]
    string_feature = one_hot(string_data)
    numerical_feature = normalize(numerical_data)
    string_feature.index = numerical_feature.index
    data_pool = pd.concat([numerical_feature, string_feature, label], axis=1)
    process_data = {}
    treatment = {0: [], 1: []}

    fin_s[fin_s["trust"].isna()]
    # fin_user = set(fin_s[~fin_s["trust"].isna()]["workerId"])
    mid_user = set(mid_s[~mid_s["trust"].isna()]["workerId"])
    ini_user = set(ini_s[~ini_s["trust"].isna()]["workerId"])
    user = set(data[~data["globalId"].isna()]["workerId"])
    user = list(user & mid_user & ini_user)

    for i in user:
        u = human()
        interaction = data.query("workerId==@i")
        if len(interaction.query("globalId==-1")["attentionCorrect"]) == 1:
            if interaction.query("globalId==-1")["attentionCorrect"].item() == 1:
                interaction = interaction[interaction["globalId"] != -1]
                interaction = interaction.sort_values(by=["taskId"])
                input_task = pd.DataFrame(columns=data_pool.columns)
                #     ml_effect = (interaction["income"] == interaction["mlPrediction"]).apply(lambda x:1 if x else -1)
                for k in range(21):
                    instance = interaction.query("taskId==@k")
                    globalId = instance["globalId"]
                    if not globalId.empty:
                        globalId = int(globalId.iloc[0].item())
                        task_feature = data_pool[data_pool["id"] == globalId]
                        input_task.loc[k] = task_feature.loc[task_feature.index.item()]
                ml_effect = (input_task["income"] == input_task["ml_pred"]).apply(lambda x: 1 if x else -1)
                ml_effect = pd.DataFrame({'ml_effect': [1] + list(ml_effect.values)[:19]})
                human_decision = pd.DataFrame({'decision': (
                            interaction["mlPrediction"] == interaction["prediction"]).apply(lambda x: 1 if x else 0)})
                human_correct = pd.DataFrame(
                    {'correct': (interaction["prediction"] == interaction["income"]).apply(lambda x: 1 if x else 0)})
                if len(ml_effect) == 20 and len(input_task) == 20 and len(human_decision) == 20:
                    ml_effect.index = input_task.index
                    human_decision.index = input_task.index
                    human_correct.index = input_task.index
                    input_task = pd.concat([input_task, ml_effect, human_decision, human_correct], axis=1)
                    u.data = input_task
                    treatment[list(interaction["treatment"])[0]].append(i)
                    if len(mid_s.query("workerId==@i")["trust"]) == 1 and len(
                            ini_s.query("workerId==@i")["trust"]) == 1:
                        ini_user = ini_s.query("workerId==@i")["trust"].item()
                        mid_user = mid_s.query("workerId==@i")["trust"].item()
                        pos = mid_s.query("workerId==@i")["taskId"].item()
                        fin_user = fin_s.query("workerId==@i")[["gender", "age", "education", "programming"]]
                        # modify gender feature
                        u.ini_trust = ini_user
                        u.mid_trust = [mid_user, pos]
                        u.demo = fin_user
                        process_data[i] = u

    from sklearn.model_selection import RepeatedKFold
    import random
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
    from sklearn.linear_model import Perceptron, LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import log_loss

    random_state = 2022


    def fivefold(index):
        random.shuffle(index)
        interval = len(index) // 5
        l = []
        for i in range(5):
            if i == 4:
                l.append(index[i * interval:])
            else:
                l.append(index[i * interval:(i + 1) * interval])
        fold = []
        for i in range(5):
            test = l[i]
            train = [l[k] for k in range(5) if k != i]
            e = []
            for t in train:
                e += t
            train = e
            fold.append([train, test])
        return fold


    human_data = [v for k, v in process_data.items()]
    index = [i for i in range(len(process_data))]
    fold = fivefold(index)
    print(human_data[0].data.columns)
    feature_columns = ['age', 'education.num', 'hours.per.week', 'workclass0', 'workclass1',
                       'workclass2', 'marital.status0', 'marital.status1', 'marital.status2',
                       'occupation0', 'occupation1', 'occupation2', 'occupation3',
                       'occupation4', 'occupation5', 'occupation6', 'occupation7',
                       'occupation8', 'occupation9', 'occupation10', 'occupation11', 'sex0',
                       'sex1', 'ml_effect']
    F1_w = []
    F1_ma = []
    AUC = []
    Precision = []
    Recall = []
    Confusion = []
    FPR = []
    FNR = []
    Neglog = []

    for train, test in fold:
        train_data = [human_data[i].data for i in train]
        test_data = [human_data[i].data for i in test]
        train_X = [d[feature_columns].values for d in train_data]
        train_Y = [d["decision"].values for d in train_data]
        test_X = [d[feature_columns].values for d in test_data]
        test_Y = [d["decision"].values for d in test_data]

        train_dataset = human_dataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_dataset = human_dataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        model = RNN(train_X[0].shape[1])

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
        init_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        for e in range(50):
            train_model(model, train_loader=train_loader, optimizer=optimizer)
            if e >= 20:
                test_model(model, test_loader=test_loader)



