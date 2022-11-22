import numpy as np
import pandas as pd
from hmm import UnSupervisedIOHMM
from hmm import OLS,DiscreteMNL, CrossEntropyMNL


if __name__=='__main__':
    # preprocess data
    import re
    import copy
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn import preprocessing
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd
    import random


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
        human_decision = pd.DataFrame(
            {'decision': (interaction["mlPrediction"] == interaction["prediction"]).apply(lambda x: 1 if x else 0)})
        human_correct = pd.DataFrame(
            {'correct': (interaction["prediction"] == interaction["income"]).apply(lambda x: 1 if x else 0)})
        if len(ml_effect) == 20 and len(input_task) == 20 and len(human_decision) == 20:
            ml_effect.index = input_task.index
            human_decision.index = input_task.index
            human_correct.index = input_task.index
            input_task = pd.concat([input_task, ml_effect, human_decision, human_correct], axis=1)
            u.data = input_task
            treatment[list(interaction["treatment"])[0]].append(i)
            if len(mid_s.query("workerId==@i")["trust"]) == 1 and len(ini_s.query("workerId==@i")["trust"]) == 1:
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

    from hmm import UnSupervisedIOHMM, SemiSupervisedIOHMM
    from hmm import OLS, DiscreteMNL, CrossEntropyMNL

    train_data = [i.data[:10] for i in human_data]
    test_data = [i.data[10:] for i in human_data]
    train_X = [d[feature_columns + ["decision"]] for d in train_data]
    test_X = [d[feature_columns] for d in test_data]
    test_Y = [d["decision"] for d in test_data]
    uhmm = UnSupervisedIOHMM(num_states=5, max_EM_iter=1, EM_tol=1e-6)
    uhmm.set_models(model_emissions=[DiscreteMNL(alpha=1, reg_method='l2')],
                    model_transition=CrossEntropyMNL(alpha=4, solver='lbfgs', reg_method='l2'),
                    model_initial=CrossEntropyMNL(alpha=4, solver='lbfgs', reg_method='l2'))
    uhmm.set_inputs(covariates_initial=feature_columns, covariates_transition=feature_columns,
                    covariates_emissions=[feature_columns])
    uhmm.set_outputs([['decision']])
    uhmm.set_data(train_X)
    uhmm.train()
    auc = []
    f1_w = []
    f1_ma = []
    precision = []
    recall = []
    fpr = []
    fnr = []
    neglog = []
    for i in range(len(test_X)):
        d = pd.concat([train_X[i][feature_columns], test_X[i]], axis=0)
        preds, probs, states = uhmm.predict_new_user(d)
        probs = probs[10:]
        probs_auc = [i[0, 1] for i in probs]
        preds = preds[10:]
        f1_w.append(f1_score(test_Y[i], preds, average='weighted'))
        f1_ma.append(f1_score(test_Y[i], preds, average='macro'))
        precision.append(precision_score(test_Y[i], preds, average='macro'))
        recall.append(recall_score(test_Y[i], preds, average='macro'))
        probs = np.concatenate([i for i in probs], axis=0)
        if sum(test_Y[i]) == 10:
            test_Y[i].loc[21] = 0
            probs_auc.append(0.2)
            preds.append(0)
            probs = np.concatenate((probs, np.array([[0.8, 0.2]])), axis=0)
        if sum(test_Y[i]) == 0:
            test_Y[i].loc[21] = 1
            probs_auc.append(0.8)
            preds.append(1)
            probs = np.concatenate((probs, np.array([[0.2, 0.8]])), axis=0)
        neglog.append(len(test_Y[i]) * log_loss(test_Y[i], probs))
        c = confusion_matrix(test_Y[i], preds)
        c = c.astype('float') / c.sum(axis=1)[:, np.newaxis]
        fpr.append(c[0, 1])
        fnr.append(c[1, 0])
        auc.append(roc_auc_score(test_Y[i], probs_auc))
    F1_w.append(f1_w)
    F1_ma.append(f1_ma)
    AUC.append(auc)
    Precision.append(precision)
    Recall.append(recall)
    FPR.append(fpr)
    FNR.append(fnr)
    Neglog.append(neglog)




