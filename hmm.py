

from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import object
from copy import deepcopy
import logging
import os
import warnings

import numpy as np

from forward_backward import forward_backward
from linear_models import (GLM, OLS, DiscreteMNL, CrossEntropyMNL)

warnings.simplefilter("ignore")
np.random.seed(0)
EPS = np.finfo(float).eps


def correct_helper(action,ml_pred, label):
    if action == 1:
        return 1 if ml_pred == label else -1
    else:
        return -1 if ml_pred == label else 1

class LinearModelLoader(object):
  
    GLM = GLM
    OLS = OLS
    DiscreteMNL = DiscreteMNL
    CrossEntropyMNL = CrossEntropyMNL


class Base(object):
  

    def __init__(self, num_states=2):
        """
        Constructor
        Parameters
        ----------
        num_states: the number of hidden states
        """
        self.dfs_logStates = None
        self.predict_transitions_all_sequence = None
        self.predict_transitions = None
        self.predict_initials_all_sequence = None
        self.predict_initials = None
        self.predict_dfs_log_States = None
        self.num_states = num_states
        self.trained = False

    def set_models(self, model_decisions,
                   model_initial=CrossEntropyMNL(),
                   model_transition=CrossEntropyMNL(), trained=False):
    
        if trained:
            self.model_initial = model_initial
            self.model_transition = model_transition
            self.model_decisions = model_decisions
            self.trained = True
        else:
            self.model_initial = model_initial
            self.model_transition = [deepcopy(model_initial) for _ in range(self.num_states)]
            self.model_decisions = [deepcopy(model_decisions) for _ in range(self.num_states)]

    def set_inputs(self, covariates_initial, covariates_transition, covariates_emissions):
        
        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions

    def set_outputs(self, responses_emissions):
       
        self.responses_emissions = responses_emissions
        self.num_emissions = len(responses_emissions)

    def set_data(self, dfs):
     
        raise NotImplementedError

    def _initialize(self, with_randomness=True):
        

        def _initialize_log_gamma(df, log_state):
        
            log_gamma = np.log(np.zeros((df.shape[0], self.num_states))+1e-12)
            for time_stamp in log_state:
                log_gamma[time_stamp, :] = log_state[time_stamp]
            return log_gamma

        def _initialize_log_epsilon(df, log_state):
        
            log_epsilon = np.log(np.zeros((df.shape[0] - 1, self.num_states, self.num_states))+1e-12)
            for time_stamp in log_state:
                if time_stamp + 1 in log_state:
                    # actually should find the index of 1
                    st = int(np.argmax(log_state[time_stamp]))
                    log_epsilon[time_stamp, st, :] = log_state[time_stamp + 1]
            return log_epsilon

        # initialize log_gammas
        self.log_gammas = [_initialize_log_gamma(df, log_state)
                           for df, log_state in self.dfs_logStates]
        # initialize log_epsilons
        self.log_epsilons = [_initialize_log_epsilon(df, log_state)
                             for df, log_state in self.dfs_logStates]
        if with_randomness:
            for st in range(self.num_states):
                if np.exp(np.hstack([lg[:, st] for lg in self.log_gammas])).sum() < EPS:
                    # there is no any sample associated with this state
                    for lg in self.log_gammas:
                        lg[:, st] = np.random.rand(lg.shape[0])
            for st in range(self.num_states):
                if np.exp(np.vstack([le[:, st, :] for le in self.log_epsilons])).sum() < EPS:
                    # there is no any sample associated with this state
                    for le in self.log_epsilons:
                        le[:, st, :] = np.random.rand(le.shape[0], self.num_states)

        # initialize log_likelihood
        self.log_likelihoods = [-np.Infinity for _ in range(self.num_seqs)]
        self.log_likelihood = -np.Infinity

        # initialize input/output covariates
        self.inp_initials = [np.array(df[self.covariates_initial].iloc[0]).reshape(
            1, -1).astype('float64') for df, log_state in self.dfs_logStates]

        self.inp_initials_all_sequences = np.vstack(self.inp_initials)

        self.inp_transitions = [np.array(df[self.covariates_transition].iloc[1:]).astype(
            'float64') for df, log_state in self.dfs_logStates]

        self.inp_transitions_all_sequences = np.vstack(self.inp_transitions)

        self.inp_emissions = [[np.array(df[cov]).astype('float64') for
                               cov in self.covariates_emissions]
                              for df, log_state in self.dfs_logStates]
        self.inp_emissions_all_sequences = [np.vstack([seq[emis] for
                                                       seq in self.inp_emissions]) for
                                            emis in range(self.num_emissions)]
        self.out_emissions = [[np.array(df[res]) for
                               res in self.responses_emissions]
                              for df, log_state in self.dfs_logStates]

        self.out_emissions_all_sequences = [np.vstack([seq[emis] for
                                                       seq in self.out_emissions]) for
                                            emis in range(self.num_emissions)]

    def viterbi(self):
        """
        The viterbi algorithm (dynamic programming):
        1) compute \delta and \psi
        2) backward search optimal state
        :return: the most probable hidden state
        """
        all_hidden_states = []
        for seq in range(self.predict_num):
            hidden_states = []
            n_records = self.predict_dfs_log_States[seq][0].shape[0]
            prob_initial = np.exp(self.model_initial.predict_log_proba(
                self.predict_initials[seq]).reshape(self.num_states, ))
            prob_transition = np.zeros((n_records - 1, self.num_states, self.num_states))

            for st in range(self.num_states):
                prob_transition[:, st, :] = np.exp(self.model_transition[st].predict_log_proba(
                    self.predict_transitions[seq]))

            Ey = np.zeros((n_records, self.num_states))

            for emis in range(self.num_emissions):
                model_collection = [models[emis] for models in self.model_decisions]
                Ey += np.exp(np.vstack([model.loglike_per_sample(
                    np.array(self.predict_emissions[seq][emis]).astype('float64'),
                    np.array(self.predict_out_emissions[seq][emis])) for model in model_collection]).T)

            psi = np.zeros((n_records, self.num_states))
            delta = np.zeros((n_records, self.num_states))
            for i in range(n_records):
                if i == 0:
                    delta[0, :] = prob_initial * Ey[0, :]
                    psi[0, :] = 0
                else:
                    delta[i, :] = np.max(delta[i - 1, :] * prob_transition[i - 1, :, :].T, axis=1) * Ey[i, :]
                    psi[i, :] = np.argmax(delta[i - 1, :] * prob_transition[i - 1, :, :].T, axis=1)
            ##backward
            for i in range(n_records):
                timestamp = n_records - i
                if i == 0:
                    h_i = np.argmax(delta[delta.shape[0] - 1, :])
                else:
                    h_i = int(psi[timestamp, int(h_i)])
                hidden_states.append(h_i)
            hidden_states.reverse()
            all_hidden_states.append(hidden_states)
        return all_hidden_states

   

    def generate_synthetic_data(self, data):
        assert len(data) > 0
        data["correct_effect"] = [1 for i in range(len(data))]
        predictions = []
        states = []
        probs = []
        i_data = np.array(data[self.covariates_initial][0]).astype('float64').reshape(1,-1)
        ### maybe add the initial estimate of AI correctness
        i_data = np.concatenate([i_data,[1]],1)
        input_emission = np.array(data[self.covariates_emissions[0]]).astype('float64')

        prob_initial = np.exp(self.model_initial.predict_log_proba(i_data).reshape(self.num_states, ))
        model_collection = [models[self.num_emissions - 1] for models in self.model_decisions]
        initial_state = np.argmax(prob_initial)

        prob = np.exp(model_collection[initial_state].predict_log_proba(input_emission[0,:].reshape(1,-1)))
        pred = np.argmax(prob)
        predictions.append(pred)
        probs.append(prob)
        states.append(initial_state)
        past_state = initial_state

        data["correct_effect"].iloc[1] =  correct_helper(pred, data["ml_pred"].iloc[0], data["income"].iloc[0])
        if data.shape[0] > 1:
            for i in range(1,data.shape[0]):
                t_data = np.array(data[self.covariates_transition].iloc[i]).astype('float64') .reshape(1, -1)
                e_data = np.array(data[self.covariates_emissions[0]].iloc[i]).astype('float64').reshape(1, -1)
                test_probability = np.exp(
                    self.model_transition[past_state].predict_log_proba(t_data))
                current_state = np.argmax(test_probability)
                # current_state = np.random.choice([0,1], p=test_probability.reshape(3))
                # test_probability =
                prob = np.exp(model_collection[current_state].predict_log_proba(e_data))
                pred = np.argmax(prob)
                predictions.append(pred)
                probs.append(prob)
                states.append(current_state)
                past_state = current_state
                if i < data.shape[0]-1:
                    data["correct_effect"].iloc[i+1] = correct_helper(pred, data["ml_pred"].iloc[i], data["income"].iloc[i])

        return predictions, probs, states




    def predict_new_user(self, test_data):
        assert len(test_data) > 0
        predictions = []
        states = []
        probs = []
        i_data = np.array(test_data[self.covariates_initial].iloc[0]).astype('float64').reshape(1, -1)

        input_transition = np.array(test_data[self.covariates_transition].iloc[1:]).astype('float64') if test_data.shape[0] > 1 else None
        input_emission = np.array(test_data[self.covariates_emissions[0]]).astype('float64')
        prob_initial = np.exp(self.model_initial.predict_log_proba(
            i_data).reshape(self.num_states, ))
        model_collection = [models[self.num_emissions - 1] for models in self.model_decisions]
        initial_state = np.argmax(prob_initial)

        prob = np.exp(model_collection[initial_state].predict_log_proba(input_emission[0,:].reshape(1,-1)))
        pred = np.argmax(prob)
        predictions.append(pred)
        probs.append(prob)

        states.append(initial_state)
        past_state = initial_state
        if test_data.shape[0] > 1:
            for i in range(test_data.shape[0] - 1):
                t_data = input_transition[i, :].reshape(1, -1)
                e_data = input_emission[i + 1, :].reshape(1, -1)
                test_probability = np.exp(
                    self.model_transition[past_state].predict_log_proba(t_data))
                current_state = np.argmax(test_probability)
                # current_state = np.random.choice([0,1], p=test_probability.reshape(3))
                # test_probability =
                prob = np.exp(model_collection[current_state].predict_log_proba(e_data))
                pred = np.argmax(prob)
                predictions.append(pred)
                probs.append(prob)
                states.append(current_state)
                past_state = current_state

        return predictions, probs, states

    def predict_old_user(self, past_data, test_data):


        self.predict_dfs_log_States = [[past_data.iloc[x], {}] for x in range(len(past_data))]
        self.predict_num = len(self.predict_dfs_log_States)

        self.predict_initials = [np.array(df[self.covariates_initial].iloc[0]).reshape(1, -1).astype('float64') for
                                 df, logstate in self.predict_dfs_log_States]
        self.predict_transitions = [np.array(df[self.covariates_transition].iloc[1:]).astype('float64') for df, logstate
                                    in self.predict_dfs_log_States]

        self.predict_emissions = [[np.array(df[cov]).astype('float64') for cov in self.covariates_emissions] for
                                  df, logstate in self.predict_dfs_log_States]
        self.predict_out_emissions = [[np.array(df[res]) for res in self.responses_emissions] for df, log_state in
                                      self.predict_dfs_log_States]
        hidden_sequences = self.viterbi()
        predictions = []
        probabs = []
        model_collection = [models[self.num_emissions - 1] for models in self.model_decisions]

        for j in range(self.predict_num):
            prediction = []
            past_state = hidden_sequences[j][-1]
            input_transition = np.array(test_data[j][self.covariates_transition]).astype('float64')
            input_emission =  np.array(test_data[j][self.covariates_emissions[0]]).astype('float64')
            for i in range(input_emission.shape[0]):
                t_data = input_transition[i, :].reshape(1, -1)
                e_data = input_emission[i, :].reshape(1, -1)
                test_probability = np.exp(self.model_transition[past_state].predict_log_proba(t_data))
                # print(test_probability)
                current_state = np.argmax(test_probability)
                # test_probability =
                # print(np.exp(model_collection[current_state].predict_log_proba(e_data)))
                # print(prediction)
                prob = np.exp(model_collection[current_state].predict_log_proba(e_data))[0]
                # print(prob)
                pred = np.argmax(prob)
                hidden_sequences[j].append(current_state)
                prediction.append(pred)
                probabs.append(prob)
                past_state = current_state
            predictions.append(prediction)
            return predictions, probabs, hidden_sequences

    def E_step(self):
       
        self.log_gammas = []
        self.log_epsilons = []
        self.log_likelihoods = []
        for seq in range(self.num_seqs):
            n_records = self.dfs_logStates[seq][0].shape[0]
            # initial probability
            log_prob_initial = self.model_initial.predict_log_proba(
                self.inp_initials[seq]).reshape(self.num_states, )
            # transition probability
            log_prob_transition = np.zeros((n_records - 1, self.num_states, self.num_states))
            for st in range(self.num_states):
                log_prob_transition[:, st, :] = self.model_transition[st].predict_log_proba(
                    self.inp_transitions[seq])

            assert log_prob_transition.shape == (n_records - 1, self.num_states, self.num_states)
            # emission probability
            log_Ey = np.zeros((n_records, self.num_states))
            for emis in range(self.num_emissions):
                model_collection = [models[emis] for models in self.model_decisions]
                log_Ey += np.vstack([model.loglike_per_sample(
                    np.array(self.inp_emissions[seq][emis]).astype('float64'),
                    np.array(self.out_emissions[seq][emis])) for model in model_collection]).T
            # forward backward to calculate posterior
            log_gamma, log_epsilon, log_likelihood = forward_backward(
                log_prob_initial, log_prob_transition, log_Ey, self.dfs_logStates[seq][1])
            self.log_gammas.append(log_gamma)
            self.log_epsilons.append(log_epsilon)
            self.log_likelihoods.append(log_likelihood)
        self.log_likelihood = sum(self.log_likelihoods)

    def M_step(self):
    

        # optimize initial model
        X = self.inp_initials_all_sequences
        Y = np.exp(np.vstack([lg[0, :].reshape(1, -1) for lg in self.log_gammas]))
        self.model_initial.fit(X, Y)

        # optimize transition models
        X = self.inp_transitions_all_sequences
        for st in range(self.num_states):
            Y = np.exp(np.vstack([eps[:, st, :] for eps in self.log_epsilons]))
            self.model_transition[st].fit(X, Y)

        # optimize emission models
        for emis in range(self.num_emissions):
            X = self.inp_emissions_all_sequences[emis]
            Y = self.out_emissions_all_sequences[emis]
            for st in range(self.num_states):
                sample_weight = np.exp(np.hstack([lg[:, st] for lg in self.log_gammas]))
                self.model_decisions[st][emis].fit(X, Y, sample_weight=sample_weight)

    def train(self):
 
        for it in range(self.max_EM_iter):
            log_likelihood_prev = self.log_likelihood
            self.M_step()
            self.E_step()
            logging.info('log likelihood of iteration {0}: {1:.4f}'.format(it, self.log_likelihood))
            if abs(self.log_likelihood - log_likelihood_prev) < self.EM_tol:
                break
        self.trained = True

    def to_json(self, path):
        
        json_dict = {
            'data_type': self.__class__.__name__,
            'properties': {
                'num_states': self.num_states,
                'covariates_initial': self.covariates_initial,
                'covariates_transition': self.covariates_transition,
                'covariates_emissions': self.covariates_emissions,
                'responses_emissions': self.responses_emissions,
                'model_initial': self.model_initial.to_json(
                    path=os.path.join(path, 'model_initial')),
                'model_transition': [self.model_transition[st].to_json(
                    path=os.path.join(path, 'model_transition', 'state_{}'.format(st))) for
                    st in range(self.num_states)],
                'model_decisions': [[self.model_decisions[st][emis].to_json(
                    path=os.path.join(
                        path, 'model_decisions', 'state_{}'.format(st), 'emission_{}'.format(emis))
                ) for emis in range(self.num_emissions)] for st in range(self.num_states)]
            }
        }
        return json_dict

    @classmethod
    def _from_setup(
            cls, json_dict, num_states,
            model_initial, model_transition, model_decisions,
            covariates_initial, covariates_transition, covariates_emissions,
            responses_emissions, trained):
       
        model = cls(num_states=num_states)
        model.set_models(
            model_initial=model_initial,
            model_transition=model_transition,
            model_decisions=model_decisions,
            trained=trained)
        model.set_inputs(covariates_initial=covariates_initial,
                         covariates_transition=covariates_transition,
                         covariates_emissions=covariates_emissions)
        model.set_outputs(responses_emissions=responses_emissions)
        return model

    @classmethod
    def from_config(cls, json_dict):
    
        return cls._from_setup(
            json_dict,
            num_states=json_dict['properties']['num_states'],
            model_initial=getattr(
                LinearModelLoader, json_dict['properties']['model_initial']['data_type'])(
                **json_dict['properties']['model_initial']['properties']),
            model_transition=getattr(
                LinearModelLoader, json_dict['properties']['model_transition']['data_type'])(
                **json_dict['properties']['model_transition']['properties']),
            model_decisions=[getattr(
                LinearModelLoader, model_emission['data_type'])(**model_emission['properties'])
                             for model_emission in json_dict['properties']['model_decisions']],
            covariates_initial=json_dict['properties']['covariates_initial'],
            covariates_transition=json_dict['properties']['covariates_transition'],
            covariates_emissions=json_dict['properties']['covariates_emissions'],
            responses_emissions=json_dict['properties']['responses_emissions'],
            trained=False)

    @classmethod
    def from_json(cls, json_dict):
       
        return cls._from_setup(
            json_dict,
            num_states=json_dict['properties']['num_states'],
            model_initial=getattr(
                LinearModelLoader, json_dict['properties']['model_initial']['data_type']).from_json(
                json_dict['properties']['model_initial']),
            model_transition=[getattr(
                LinearModelLoader, model_transition_json['data_type']
            ).from_json(model_transition_json) for
                              model_transition_json in json_dict['properties']['model_transition']],
            model_decisions=[[getattr(
                LinearModelLoader,model_decisions_json['data_type']
            ).from_json(model_decisions_json) for model_emission_json in model_decisions_json] for
                             model_decisions_json in json_dict['properties']['model_decisions']],
            covariates_initial=json_dict['properties']['covariates_initial'],
            covariates_transition=json_dict['properties']['covariates_transition'],
            covariates_emissions=json_dict['properties']['covariates_emissions'],
            responses_emissions=json_dict['properties']['responses_emissions'],
            trained=True)


class Trust_hmm(Base):
  

    def __init__(self, num_states=2, EM_tol=1e-4, max_EM_iter=100):
       
        super(Trust_hmm, self).__init__(num_states=num_states)
        self.EM_tol = EM_tol
        self.max_EM_iter = max_EM_iter

    def set_data(self, dfs):
        
        assert all([df.shape[0] > 0 for df in dfs])
        self.num_seqs = len(dfs)
        self.dfs_logStates = [[x, {}] for x in dfs]
        self._initialize(with_randomness=True)

    def to_json(self, path):
       
        json_dict = super(Trust_hmm, self).to_json(path)
        json_dict['properties'].update(
            {
                'EM_tol': self.EM_tol,
                'max_EM_iter': self.max_EM_iter,
            }
        )
        return json_dict

    @classmethod
    def _from_setup(
            cls, json_dict, num_states,
            model_initial, model_transition, model_decisions,
            covariates_initial, covariates_transition, covariates_emissions,
            responses_emissions, trained):
       
        model = cls(num_states=num_states,
                    EM_tol=json_dict['properties']['EM_tol'],
                    max_EM_iter=json_dict['properties']['max_EM_iter'])
        model.set_models(
            model_initial=model_initial,
            model_transition=model_transition,
            model_decisions=model_decisions,
            trained=trained)
        model.set_inputs(covariates_initial=covariates_initial,
                         covariates_transition=covariates_transition,
                         covariates_emissions=covariates_emissions)
        model.set_outputs(responses_emissions=responses_emissions)
        return model


