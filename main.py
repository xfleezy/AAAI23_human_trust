import numpy as np
import pandas as pd
from hmm import UnSupervisedIOHMM
from hmm import OLS,DiscreteMNL, CrossEntropyMNL
from sklearn.model_selection import RepeatedKFold
import random
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
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

from hmm import UnSupervisedIOHMM
from hmm import OLS, DiscreteMNL, CrossEntropyMNL

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



def generate_hmm_test(human_data, feature_columns, test):
    low_risk = {"test_data":[], "test_Y":[],"test_trust":[]}
    high_risk = {"test_data":[], "test_Y":[],"test_trust":[]}
    for i in test:
        if human_data[i].data["treatment"][0] == 0:
            low_risk["test_data"].append(human_data[i].data[feature_columns])
            low_risk["test_Y"].append(human_data[i].data["decision"])
            low_risk["test_trust"].append([human_data[i].ini_trust, human_data[i].mid_trust])
        else:
            high_risk["test_data"].append(human_data[i].data[feature_columns])
            high_risk["test_Y"].append(human_data[i].data["decision"])
            high_risk["test_trust"].append([human_data[i].ini_trust, human_data[i].mid_trust])
    return low_risk, high_risk


if __name__=='__main__':
 


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


    ini_s = pd.read_json("data/initialSurvey_0503_01.json", lines=True)
    mid_s = pd.read_json("data/misSurvey_0503_01.json", lines=True)
    fin_s = pd.read_json("data/surveys_0503_01.json", lines=True)
    data = pd.read_json("data/pilotStudy_0503_01.json", lines=True)
    data_pool = pd.read_csv("data/sample_data_0329.csv")

    for i in data_pool:
        data_pool.loc[(data_pool['age'] > 16) & (data_pool['age'] <= 25), 'age'] = 1
        data_pool.loc[(data_pool['age'] > 25) & (data_pool['age'] <= 32), 'age'] = 2
        data_pool.loc[(data_pool['age'] > 32) & (data_pool['age'] <= 40), 'age'] = 3
        data_pool.loc[(data_pool['age'] > 40) & (data_pool['age'] <= 50), 'age'] = 4
        data_pool.loc[data_pool['age'] > 50, 'age'] = 5
    
    label = data_pool[["income","ml_pred","id"]]
    data_pool = data_pool.drop(["income","ml_pred","id"], axis=1)
    numerical_var = [c for c in data_pool.columns if data_pool[c].dtype !=object]
    string_var = [c for c in data_pool.columns if data_pool[c].dtype == object]
    data_pool = data_pool[data_pool.columns]
    numerical_data = data_pool[numerical_var]
    string_data = data_pool[string_var]
    string_feature =one_hot(string_data)
    numerical_feature = normalize(numerical_data)
    string_feature.index = numerical_feature.index
    data_pool = pd.concat([numerical_feature, string_feature,label],axis=1)
    process_data = {}
    # treatment = {0:[],1:[]}
    demo_information = fin_s[["gender","age","education","programming"]]
    numerical_var = [c for c in demo_information.columns if demo_information[c].dtype != object]
    # string_var = [c for c in demo_information.columns if demo_information[c].dtype == object]
    string_var = [c for c in demo_information.columns]
    numerical_data = demo_information[numerical_var]
    string_data = demo_information[string_var]
    string_feature =one_hot(string_data)
    numerical_feature = normalize(numerical_data)
    string_feature.index = numerical_feature.index
    # demo_information = pd.concat([numerical_feature, string_feature],axis=1)
    demo_information = string_feature
    # print(demo_information.columns)
    fin_s = fin_s.drop(["gender","age","education","programming"],axis=1)
    fin_s = pd.concat([fin_s,demo_information],axis=1)
    fin_s = fin_s.rename(columns={"age": "age1"})
    fin_s[fin_s["trust"].isna()]
    # fin_user = set(fin_s[~fin_s["trust"].isna()]["workerId"])
    mid_user = set(mid_s[~mid_s["trust"].isna()]["workerId"])
    ini_user = set(ini_s[~ini_s["trust"].isna()]["workerId"])
    user = set(data[~data["globalId"].isna()]["workerId"])
    user = list(user & mid_user & ini_user)
    num = {0:0,1:0}

    t = []
    r = []
    c = []
    for i in user:
        u = human()
        interaction = data.query("workerId==@i")
        if len(interaction.query("globalId==-1")["attentionCorrect"]) == 1:
            if interaction.query("globalId==-1")["attentionCorrect"].item() == 1:
                interaction = interaction[interaction["globalId"]!=-1]
                interaction = interaction.sort_values(by=["taskId"])
                input_task = pd.DataFrame(columns=list(data_pool.columns))

            #     ml_effect = (interaction["income"] == interaction["mlPrediction"]).apply(lambda x:1 if x else -1)
                for k in range(21):
                    instance = interaction.query("taskId==@k")
                    globalId = instance["globalId"]
                    if not globalId.empty:
                        globalId = int(globalId.iloc[0].item())
                        task_feature = data_pool[data_pool["id"]==globalId]
                        input_task.loc[k] = task_feature.loc[task_feature.index.item()]
                    
                ml_effect = (input_task["income"] == input_task["ml_pred"]).apply(lambda x:1 if x else -1)
                ml_effect = pd.DataFrame({'ml_effect': [1] + list(ml_effect.values)[:19]})
           
                treatment = pd.DataFrame({'treatment':interaction["treatment"]})
                human_decision = pd.DataFrame({'decision':(interaction["mlPrediction"] == interaction["prediction"]).apply(lambda x:1 if x else 0)})
                human_correct = pd.DataFrame({'correct':(interaction["prediction"] == interaction["income"]).apply(lambda x:1 if x else 0)})
                decision_effect = human_correct["correct"].apply(lambda x:1 if x==1 else -1)
                decision_effect = pd.DataFrame({'correct_effect': [1] + list(decision_effect.values)[:19]})
                last_decision = human_decision["decision"]
                last_decision = pd.DataFrame({'last_decision': [1] + list(last_decision.values)[:19]})
                ml_pred = pd.DataFrame({'ml_pred':input_task["ml_pred"]})
    #             print(input_task["ml_pred"])
                
    #             print(human_decision)
                if len(ml_effect) == 20 and len(input_task)==20 and len(human_decision) ==20:
    #                 ml_effect.index = input_task.index
    #                 human_decision.index = input_task.index
    #                 human_correct.index = input_task.index
    #                 decision_effect.index = input_task.index
    #                 last_decision.index = input_task.index
    #                 input_task = pd.concat([input_task,ml_effect,human_decision,human_correct,decision_effect,last_decision],axis=1)
    #                 u.data = input_task
    #                 treatment[list(interaction["treatment"])[0]].append(i)
                    if len(mid_s.query("workerId==@i")["trust"]) == 1 and len(ini_s.query("workerId==@i")["trust"])==1 and not fin_s.query("workerId==@i").empty:
                        ini_user = ini_s.query("workerId==@i")["trust"].item()
                        mid_user = mid_s.query("workerId==@i")["trust"].item()
                        pos = mid_s.query("workerId==@i")["taskId"].item()
                        fin_user = fin_s.query("workerId==@i")[demo_information.columns]
                        demo_feature = pd.concat([i for i in [fin_user]*20],axis=0)
                        treatment.index = input_task.index
                        num[treatment.loc[0].item()] +=1
                        demo_feature.index = input_task.index
                        ml_effect.index = input_task.index
                        ml_pred.index = ml_pred.index
                       
                        human_decision.index = input_task.index
                        human_correct.index = input_task.index
                        decision_effect.index = input_task.index
                        last_decision.index = input_task.index
                        competence =  pd.DataFrame({'competence':[ini_s.query("workerId==@i")["competence"].item()]*20})
                        c.append(ini_s.query("workerId==@i")["competence"].item())
                        t.append(ini_s.query("workerId==@i")["trust"].item())
                        r.append(ini_s.query("workerId==@i")["reliance"].item())
                        competence.index = input_task.index
                        reliance =  pd.DataFrame({'reliance': [ini_s.query("workerId==@i")["reliance"].item()]*20})
                        reliance.index = input_task.index
                        trust =  pd.DataFrame({'trust': [ini_s.query("workerId==@i")["trust"].item()]*20})
                        trust.index = input_task.index
                        input_task = pd.concat([competence,reliance, trust,ml_pred,input_task,ml_effect,human_decision,human_correct,decision_effect,last_decision,demo_feature,treatment],axis=1)
                        u.data = input_task
                        # modify gender feature
    #                     if ini_user <=2:
    #                         ini_user = 0
    #                     elif ini_user == 3:
    #                         ini_user = 1
    #                     else:
    #                         ini_user = 2
                        u.ini_trust = ini_user
                        if mid_user <=2:
                            mid_user = 0
                        elif mid_user == 3:
                            mid_user = 1
                        else:
                            mid_user = 2
                        u.mid_trust = [mid_user,pos]
                        u.demo = fin_user
                        u.data = input_task
                        process_data[i] = u
               
    human_data = [v for k,v in process_data.items()]   
    index = [i for i in range(len(process_data))]
    fold = fivefold(index)
    for j in range(5):
        feedback = ["treatment",'correct_effect','ml_effect']
        
        task = ['age', 'education.num', 'hours.per.week', 'workclass0', 'workclass1', 
               'workclass2', 'marital.status0', 'marital.status1', 'marital.status2',
               'occupation0', 'occupation1', 'occupation2', 'occupation3',
               'occupation4', 'occupation5', 'occupation6', 'occupation7',
               'occupation8', 'occupation9', 'occupation10', 'occupation11', 'sex0',
               'sex1']
        demo = ['gender0', 'gender1', 'age0', 'age1', 'age2', 'age3', 'age4', 'age5',
               'education0', 'education1', 'education2', 'education3', 'education4',
               'programming0', 'programming1', 'programming2', 'programming3']
        initial_columns = ['age', 'education.num', 'hours.per.week', 'workclass0', 'workclass1', 
               'workclass2', 'marital.status0', 'marital.status1', 'marital.status2',
               'occupation0', 'occupation1', 'occupation2', 'occupation3',
               'occupation4', 'occupation5', 'occupation6', 'occupation7',
               'occupation8', 'occupation9', 'occupation10', 'occupation11', 'sex0',
               'sex1','ml_effect','correct_effect','gender0', 'gender1', 'age0', 'age1', 'age2', 'age3', 'age4', 'age5',
               'education0', 'education1', 'education2', 'education3', 'education4',
               'programming0', 'programming1', 'programming2', 'programming3',"treatment"]
        transition_columns = ['age', 'education.num', 'hours.per.week', 'workclass0', 'workclass1', 
               'workclass2', 'marital.status0', 'marital.status1', 'marital.status2',
               'occupation0', 'occupation1', 'occupation2', 'occupation3',
               'occupation4', 'occupation5', 'occupation6', 'occupation7',
               'occupation8', 'occupation9', 'occupation10', 'occupation11', 'sex0',
               'sex1','correct_effect',"treatment"] +['ml_effect']
        emission_columns = ['age', 'education.num', 'hours.per.week', 'workclass0', 'workclass1', 
               'workclass2', 'marital.status0', 'marital.status1', 'marital.status2',
               'occupation0', 'occupation1', 'occupation2', 'occupation3',
               'occupation4', 'occupation5', 'occupation6', 'occupation7',
               'occupation8', 'occupation9', 'occupation10', 'occupation11', 'sex0',
               'sex1','ml_effect','correct_effect',"treatment"]
        
        feature_columns = ['age', 'education.num', 'hours.per.week', 'workclass0', 'workclass1', 
           'workclass2', 'marital.status0', 'marital.status1', 'marital.status2',
           'occupation0', 'occupation1', 'occupation2', 'occupation3',
           'occupation4','ml_effect', 'correct_effect', 'occupation5', 'occupation6', 'occupation7',
           'occupation8', 'occupation9', 'occupation10', 'occupation11', 'sex0',
           'sex1','gender0', 'gender1', 'age0', 'age1', 'age2', 'age3', 'age4', 'age5',
           'education0', 'education1', 'education2', 'education3', 'education4',
           'programming0', 'programming1', 'programming2', 'programming3','treatment',"trust","competence","reliance"]

        F1_w_low = []
        F1_ma_low = []
        F1_ma_high = []
        Test_trust_low = []
        Test_trust_high = []
        S_high = []
        S_low = []
        results_low = []
        results_high = []
        for train, test in fold:
            train_data = [human_data[i].data for i in train]
            test_data = [human_data[i].data for i in test]
            train_X = [d[feature_columns+["decision"]] for d in train_data]
            low_risk, high_risk = generate_hmm_test(human_data, feature_columns, test)
            uhmm = UnSupervisedIOHMM(num_states=3, max_EM_iter=150, EM_tol=1e-6)
            uhmm.set_models(model_emissions=[DiscreteMNL(alpha=1, reg_method='l2')],
                            model_transition=CrossEntropyMNL(alpha=4, solver='lbfgs', reg_method='l2'),
                            model_initial=CrossEntropyMNL(alpha=4, solver='lbfgs', reg_method='l2'))
            uhmm.set_inputs(covariates_initial=emission_columns,covariates_transition=emission_columns,
                            covariates_emissions=[emission_columns])
            uhmm.set_outputs([['decision']])
            uhmm.set_data(train_X)
            uhmm.train()
            f1_w_low = []
            f1_ma_low = []
            State_low = []
            State_high = []
            test_X_low, test_Y_low = low_risk["test_data"], low_risk["test_Y"]
            test_trust_low = low_risk["test_trust"]
            Test_trust_low.append(test_trust_low)
         
            result_low = []
            result_high = []
            for i in range(len(test_X_low)):
                preds, probs, states = uhmm.predict_new_user(test_X_low[i])

#                 labels_low.extend(test_Y_low[i].to_list())
                probs_auc = [i[0,1] for i in probs]
#                 pred_ys_low.extend(probs_auc)
                f1_w_low.append(f1_score(test_Y_low[i], preds, average='weighted'))
                f1_ma_low.append(f1_score(test_Y_low[i], preds, average='macro'))
                a = {"data":test_X_low[i],"label":test_Y_low[i],"pred":preds,"probs":probs}
                result_low.append(a)
                probs = np.concatenate([i for i in probs], axis=0)
               
    #    
                State_low.append(states)

            F1_w_low.append(f1_w_low)
            F1_ma_low.append(f1_ma_low)
            S_low.append(State_low)

            f1_w_high = []
            f1_ma_high = []
            test_X_high, test_Y_high = high_risk["test_data"], high_risk["test_Y"]
            test_trust_high = high_risk["test_trust"]
            Test_trust_high.append(test_trust_high)
            for i in range(len(test_X_high)):
                preds, probs, states = uhmm.predict_new_user(test_X_high[i])

                f1_w_high.append(f1_score(test_Y_high[i], preds, average='weighted'))
                f1_ma_high.append(f1_score(test_Y_high[i], preds, average='macro'))
                a = {"data":test_X_high[i],"label":test_Y_high[i],"pred":preds,"probs":probs}
                result_high.append(a)
                probs = np.concatenate([i for i in probs], axis=0)
          
                State_high.append(states)


            F1_ma_high.append(f1_ma_high)
            S_high.append(State_high)
            results_low.append(result_low)
            results_high.append(result_high)
      

    print("performance {} : ---------------------->s".format(j))
    print("method:{}, F1_ma_low:{}, std:{}".format("hmm", np.mean([sum(a)/len(a) for a in F1_ma_low]),np.std([sum(a)/len(a) for a in F1_ma_low])))
    print("method:{}, F1_ma_high:{}, std:{}".format("hmm", np.mean([sum(a)/len(a) for a in F1_ma_high]),np.std([sum(a)/len(a) for a in F1_ma_high])))
#         print("method:{}, R_low:{}, std