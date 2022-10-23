import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference import Inference
from model.inference_utils import reset_layer, forward_network, forward_network_batch, get_state_abstraction
from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class InferenceCMI(Inference):
    def __init__(self, encoder, params):
        self.cmi_params = params.inference_params.cmi_params
        self.init_graph(params, encoder)
        super(InferenceCMI, self).__init__(encoder, params)
        self.causal_pred_reward_mean = 0
        self.causal_pred_reward_std = 1
        self.pred_diff_reward_std = 1

        self.init_abstraction()
        self.init_cache()
        self.reset_causal_graph_eval()

        self.update_num = 0

    def init_model(self):
        params = self.params
        cmi_params = self.cmi_params

        # model params
        continuous_state = self.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        if not self.continuous_state:
            self.feature_inner_dim = self.encoder.feature_inner_dim

        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases = nn.ParameterList()
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()
        self.generative_weights = nn.ParameterList()
        self.generative_biases = nn.ParameterList()

        # only needed for discrete state space
        self.state_feature_1st_layer_weights = nn.ParameterList()
        self.state_feature_1st_layer_biases = nn.ParameterList()
        self.generative_last_layer_weights = nn.ParameterList()
        self.generative_last_layer_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable
        # action feature extractor
        in_dim = action_dim
        for out_dim in cmi_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if continuous_state:
            in_dim = 1
            fc_dims = cmi_params.feature_fc_dims
        else:
            out_dim = cmi_params.feature_fc_dims[0]
            fc_dims = cmi_params.feature_fc_dims[1:]
            for feature_i_dim in self.feature_inner_dim:
                in_dim = feature_i_dim
                self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        for out_dim in fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim * feature_dim, 1, out_dim)))
            in_dim = out_dim

        # predictor
        in_dim = cmi_params.feature_fc_dims[-1]
        for out_dim in cmi_params.generative_fc_dims:
            self.generative_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.generative_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        if continuous_state:
            self.generative_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, 2)))
            self.generative_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, 2)))
        else:
            for feature_i_dim in self.feature_inner_dim:
                final_dim = 2 if feature_i_dim == 1 else feature_i_dim
                self.generative_last_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, final_dim)))
                self.generative_last_layer_biases.append(nn.Parameter(torch.zeros(1, 1, final_dim)))

    def reset_params(self):
        feature_dim = self.feature_dim
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(feature_dim * feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_weights, self.generative_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.generative_last_layer_weights, self.generative_last_layer_biases):
            reset_layer(w, b)

    def init_graph(self, params, encoder):
        feature_dim = encoder.feature_dim
        device = params.device
        self.CMI_threshold = self.cmi_params.CMI_threshold

        # used for masking diagonal elements
        self.diag_mask = torch.eye(feature_dim, feature_dim + 1, dtype=torch.bool, device=device)
        self.mask_CMI = torch.ones(feature_dim, feature_dim + 1, device=device) * self.CMI_threshold
        self.mask = torch.ones(feature_dim, feature_dim + 1, dtype=torch.bool, device=device)
        self.CMI_history = []

    def init_abstraction(self):
        self.abstraction_quested = False
        self.abstraction_graph = None
        self.action_children_idxes = None

    def init_cache(self):
        # cache for faster mask updates
        self.use_cache = False
        self.sa_feature_cache = None
        self.action_feature = None
        self.full_state_feature = None
        self.causal_state_feature = None

        feature_dim = self.feature_dim
        self.feature_diag_mask = torch.eye(feature_dim, dtype=torch.float32, device=self.device)
        self.feature_diag_mask = self.feature_diag_mask.view(feature_dim, feature_dim, 1, 1)

    def reset_causal_graph_eval(self):
        self.mask_update_idx = 0
        self.eval_step_CMI = torch.zeros(self.feature_dim, self.feature_dim, device=self.device)

    def extract_action_feature(self, action):
        """
        :param action: (bs, action_dim). notice that bs must be 1D
        :return: (feature_dim, 1, bs, out_dim)
        """
        action = action.unsqueeze(dim=0)                                    # (1, bs, action_dim)
        action = action.expand(self.feature_dim, -1, -1)                    # (feature_dim, bs, action_dim)
        action_feature = forward_network(action, self.action_feature_weights, self.action_feature_biases)
        return action_feature.unsqueeze(dim=1)                              # (feature_dim, 1, bs, out_dim)

    def extract_state_feature(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs must be 1D
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        feature_dim = self.feature_dim
        if self.continuous_state:
            bs = feature.shape[0]
            x = feature.transpose(0, 1)                                     # (feature_dim, bs)
            x = x.repeat(feature_dim, 1, 1)                                 # (feature_dim, feature_dim, bs)
            x = x.view(feature_dim * feature_dim, bs, 1)                    # (feature_dim * feature_dim, bs, 1)
        else:
            bs = feature[0].shape[0]
            # [(bs, feature_i_dim)] * feature_dim
            reshaped_feature = []
            for f_i in feature:
                f_i = f_i.repeat(feature_dim, 1, 1)                         # (feature_dim, bs, feature_i_dim)
                reshaped_feature.append(f_i)
            x = forward_network_batch(reshaped_feature,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases)
            x = torch.stack(x, dim=1)                                       # (feature_dim, feature_dim, bs, out_dim)
            x = x.view(feature_dim * feature_dim, *x.shape[2:])             # (feature_dim * feature_dim, bs, out_dim)

        state_feature = forward_network(x, self.state_feature_weights, self.state_feature_biases)
        state_feature = state_feature.view(feature_dim, feature_dim, bs, -1)
        return state_feature                                                # (feature_dim, feature_dim, bs, out_dim)

    def extract_masked_state_feature(self, masked_feature, full_state_feature):
        """
        :param masked_feature:
            if state space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param full_state_feature: (feature_dim, feature_dim, bs, out_dim), calculated by self.extract_state_feature()
        :return: (feature_dim, feature_dim, bs, out_dim),
            the first feature_dim is each state variable at next time step to predict, the second feature_dim are
            inputs (all current state variables) for the prediction
        """
        feature_dim = self.feature_dim

        if self.continuous_state:
            x = masked_feature.transpose(0, 1)                              # (feature_dim, bs)
            x = x.unsqueeze(dim=-1)                                         # (feature_dim, bs, 1)
        else:
            # [(1, bs, feature_i_dim)] * feature_dim
            masked_feature = [f_i.unsqueeze(dim=0) for f_i in masked_feature]
            x = forward_network_batch(masked_feature,
                                      [w[i:i+1] for i, w in enumerate(self.state_feature_1st_layer_weights)],
                                      [b[i:i+1] for i, b in enumerate(self.state_feature_1st_layer_biases)])
            x = torch.cat(x, dim=0)                                         # (feature_dim, bs, out_dim)

        idxes = [i * (feature_dim + 1) for i in range(feature_dim)]
        x = forward_network(x,
                            [w[idxes] for w in self.state_feature_weights],
                            [b[idxes] for b in self.state_feature_biases])  # (feature_dim, bs, out_dim)

        feature_diag_mask = self.feature_diag_mask                          # (feature_dim, feature_dim, 1, 1)
        masked_state_feature = x.unsqueeze(dim=0)                           # (1, feature_dim, bs, out_dim)
        masked_state_feature = full_state_feature * (1 - feature_diag_mask) + masked_state_feature * feature_diag_mask
        return masked_state_feature                                         # (feature_dim, feature_dim, bs, out_dim)

    def predict_from_sa_feature(self, sa_feature, residual_base=None, abstraction_mode=False):
        """
        predict the distribution and sample for the next step value of all state variables
        :param sa_feature: (feature_dim, bs, sa_feature_dim), global feature used for prediction,
            notice that bs can be a multi-dimensional batch size
        :param residual_base: (bs, feature_dim), residual used for continuous state variable prediction
        :param abstraction_mode: if the prediction is computed for state variables in the abstraction only.
            If True, all feature_dim in this function should be replaced by abstraction_feature_dim when indicating
            shapes of tensors.
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        if abstraction_mode:
            generative_weights = self.abstraction_generative_weights
            generative_biases = self.abstraction_generative_biases
            generative_last_layer_weights = self.abstraction_generative_last_layer_weights
            generative_last_layer_biases = self.abstraction_generative_last_layer_biases
        else:
            generative_weights, generative_biases = self.generative_weights, self.generative_biases
            generative_last_layer_weights = self.generative_last_layer_weights
            generative_last_layer_biases = self.generative_last_layer_biases

        x = forward_network(sa_feature, generative_weights, generative_biases)

        if self.continuous_state:
            x = x.permute(1, 0, 2)                                          # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)                                  # (bs, feature_dim) * 2
            return self.normal_helper(mu, residual_base, log_std)
        else:
            x = F.relu(x)                                                   # (feature_dim, bs, out_dim)
            x = [x_i.unsqueeze(dim=0) for x_i in torch.unbind(x, dim=0)]    # [(1, bs, out_dim)] * feature_dim
            x = forward_network_batch(x,
                                      generative_last_layer_weights,
                                      generative_last_layer_biases,
                                      activation=None)

            feature_inner_dim = self.feature_inner_dim
            if abstraction_mode:
                feature_inner_dim = feature_inner_dim

            dist = []
            for base_i, feature_i_inner_dim, dist_i in zip(residual_base, feature_inner_dim, x):
                dist_i = dist_i.squeeze(dim=0)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)            # (bs, 1), (bs, 1)
                    dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist

    def forward_step(self, full_feature, masked_feature, causal_feature, action, mask=None,
                     action_feature=None, full_state_feature=None):
        """
        :param full_feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, feature_i_dim)] * feature_dim
            if it is None, no need to forward it
        :param masked_feature: (bs, feature_dim) or [(bs, feature_i_dim)] * feature_dim
        :param causal_feature: (bs, feature_dim) or [(bs, feature_i_dim)] * feature_dim
        :param action: (bs, action_dim)
        :param mask: (bs, feature_dim, feature_dim + 1)
        :param action_feature: (bs, feature_dim, 1, out_dim), pre-cached value
        :param full_state_feature: (bs, feature_dim, feature_dim, out_dim), pre-cached value
        :param no_causal: not to forward causal_feature, used for training
        :param causal_only: whether to only forward causal_feature, used for curiosity reward and model-based roll-out
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        forward_full = full_feature is not None
        forward_masked = masked_feature is not None
        forward_causal = causal_feature is not None

        full_dist = masked_dist = causal_dist = None

        if action_feature is None:
            # extract features of the action
            # (feature_dim, 1, bs, out_dim)
            self.action_feature = action_feature = self.extract_action_feature(action)

        if forward_full:
            # 1. extract features of all state variables
            if full_state_feature is None:
                # (feature_dim, feature_dim, bs, out_dim)
                self.full_state_feature = full_state_feature = self.extract_state_feature(full_feature)

            # 2. extract global feature by element-wise max
            # (feature_dim, feature_dim + 1, bs, out_dim)
            full_sa_feature = torch.cat([full_state_feature, action_feature], dim=1)
            full_sa_feature, full_sa_indices = full_sa_feature.max(dim=1)           # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            full_dist = self.predict_from_sa_feature(full_sa_feature, full_feature)

        if forward_masked:
            # 1. extract features of all state variables and the action
            # (feature_dim, feature_dim, bs, out_dim)
            masked_state_feature = self.extract_masked_state_feature(masked_feature, full_state_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (feature_dim, feature_dim + 1, bs, out_dim)
            masked_sa_feature = torch.cat([masked_state_feature, action_feature], dim=1)
            mask = mask.permute(1, 2, 0)                                            # (feature_dim, feature_dim + 1, bs)
            masked_sa_feature[~mask] = float('-inf')
            masked_sa_feature, masked_sa_indices = masked_sa_feature.max(dim=1)     # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            masked_dist = self.predict_from_sa_feature(masked_sa_feature, masked_feature)

        if forward_causal:
            # 1. extract features of all state variables and the action
            causal_state_feature = self.extract_state_feature(causal_feature)

            # 2. extract global feature by element-wise max
            # mask out unused features
            # (feature_dim, feature_dim + 1, bs, out_dim)
            causal_sa_feature = torch.cat([causal_state_feature, action_feature], dim=1)
            eval_mask = self.mask.detach()                                          # (feature_dim, feature_dim + 1)
            causal_sa_feature[~eval_mask] = float('-inf')
            causal_sa_feature, causal_sa_indices = causal_sa_feature.max(dim=1)     # (feature_dim, bs, out_dim)

            # 3. predict the distribution of next time step value
            causal_dist = self.predict_from_sa_feature(causal_sa_feature, causal_feature)

        return full_dist, masked_dist, causal_dist

    def extract_action_feature_abstraction(self, action):
        """
        :param action: (bs, action_dim). notice that bs must be 1D
        :return: {action_children_idx: (1, bs, out_dim)}
        """
        num_action_children = len(self.action_children_idxes)
        action = action.unsqueeze(dim=0)                                    # (1, bs, action_dim)
        action = action.expand(num_action_children, -1, -1)                 # (num_action_children, bs, action_dim)
        # (num_action_children, bs, out_dim)
        action_feature = forward_network(action,
                                         self.abstraction_action_feature_weights,
                                         self.abstraction_action_feature_biases)
        action_feature = action_feature.unsqueeze(dim=1)                    # (num_action_children, 1, bs, out_dim)
        action_feature = torch.unbind(action_feature, dim=0)                # [(1, bs, out_dim)] * num_action_children
        action_feature_dict = {idx: action_feature_i
                               for idx, action_feature_i in zip(self.action_children_idxes, action_feature)}

        return action_feature_dict                                          # {action_children_idx: (1, bs, out_dim)}

    def extract_state_feature_abstraction(self, feature):
        """
        :param feature:
            if state space is continuous: (bs, abstraction_feature_dim).
            else: [(bs, feature_i_dim)] * abstraction_feature_dim
            notice that bs must be 1D
        :return: {state_variable_idx: (num_parent, bs, out_dim)}
        """
        if self.continuous_state:
            feature = feature.transpose(0, 1)                                   # (abstraction_feature_dim, bs)

        features = []
        for idx, parent_idxes in self.abstraction_adjacency.items():
            feature_idx = [self.abstraction_idxes.index(parent_idx) for parent_idx in parent_idxes]
            if self.continuous_state:
                x = feature[feature_idx]                                        # (num_parent, bs)
                x = x.unsqueeze(dim=-1)                                         # (num_parent, bs, 1)
                features.append(x)
            else:
                x = [feature[parent_idx] for parent_idx in feature_idx]         # [(bs, feature_i_dim)] * num_parent
                x = [x_i.unsqueeze(dim=0) for x_i in x]                         # [(1, bs, feature_i_dim)] * num_parent
                state_feature_1st_layer_weights = self.abstraction_state_feature_1st_layer_weights[idx]
                state_feature_1st_layer_biases = self.abstraction_state_feature_1st_layer_biases[idx]
                x = forward_network_batch(x,
                                          state_feature_1st_layer_weights,
                                          state_feature_1st_layer_biases)       # [(1, bs, out_dim)] * num_parent
                features.extend(x)
        features = torch.cat(features, dim=0)                                   # (total_num_parent, bs, 1)

        state_feature = forward_network(features,
                                        self.abstraction_state_feature_weights,
                                        self.abstraction_state_feature_biases)

        state_feature_dict = {}
        offset = 0
        for idx, parent_idxes in self.abstraction_adjacency.items():
            num_parents = len(parent_idxes)
            state_feature_dict[idx] = state_feature[offset:offset + num_parents]    # (num_parent, bs, out_dim)
            offset += num_parents
        return state_feature_dict

    def forward_step_abstraction(self, abstraction_feature, action):
        """
        :param abstraction_feature: if state space is continuous: (bs, abstraction_feature_dim)
            Otherwise: [(bs, feature_i_dim)] * abstraction_feature_dim
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, abstraction_feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * abstraction_feature_dim,
                each of shape (bs, feature_i_dim)
        """

        # 1. extract features of all state variables and the action
        # {action_children_idx: (1, bs, out_dim)}
        action_feature = self.extract_action_feature_abstraction(action)
        # {state_variable_idx: (num_parent, bs, out_dim)}
        state_feature = self.extract_state_feature_abstraction(abstraction_feature)

        # 2. extract global feature by element-wise max
        sa_feature = []
        for idx in self.abstraction_idxes:
            sa_feature_i = state_feature[idx]
            if idx in action_feature:
                action_feature_i = action_feature[idx]                              # (1, bs, out_dim)
                sa_feature_i = torch.cat([sa_feature_i, action_feature_i], dim=0)   # (num_parent + 1, bs, out_dim)
            sa_feature_i, _ = sa_feature_i.max(dim=0)                               # (bs, out_dim)}
            sa_feature.append(sa_feature_i)
        # (abstraction_feature_dim, bs, out_dim)
        sa_feature = torch.stack(sa_feature, dim=0)

        # 3. predict the distribution of next time step value
        dist = self.predict_from_sa_feature(sa_feature, abstraction_feature, abstraction_mode=True)

        return dist

    def forward_with_feature(self, feature, actions, mask=None,
                             forward_mode=("full", "masked", "causal"), abstraction_mode=False):
        """

        :param feature: (bs, feature_dim) if state space is continuous else [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param actions: (bs, n_pred_step, action_dim) if self.continuous_action else (bs, n_pred_step, 1)
            notice that bs can be a multi-dimensional batch size
        :param mask: (bs, feature_dim, feature_dim + 1),
            randomly generated training mask used when forwarding masked_feature
            notice that bs can be a multi-dimensional batch size
        :param forward_mode
        :param abstraction_mode: whether to only forward controllable & action-relevant state variables,
            used for model-based roll-out
        :return: a single distribution or a list of distributions depending on forward_mode,
            each distribution is of shape (bs, n_pred_step, feature_dim)
            notice that bs can be a multi-dimensional batch size
        """
        def get_dist_and_feature(prev_dist_, dist_):
            if dist_ is None:
                return None, None
            feature_ = self.sample_from_distribution(dist_)
            return dist_, feature_

        # convert feature, actions, mask to 2D tensor if bs is multi-dimensional
        reshaped = False
        bs = actions.shape[:-2]
        if len(bs) > 1:
            reshaped = True
            if isinstance(feature, Distribution):
                reshaped_mu = feature.mean.view(-1, self.feature_dim)
                reshaped_std = feature.stddev.view(-1, self.feature_dim)
                feature = Normal(reshaped_mu, reshaped_std)
            elif self.continuous_state:
                feature = feature.view(-1, self.feature_dim)
            else:
                feature = [feature_i.view(-1, self.feature_dim) for feature_i in feature]

            actions = actions.view(-1, *actions.shape[-2:])
            if mask is not None:
                mask = mask.view(-1, *mask.shape[-2:])

        # sample feature if input is a distribution
        prev_full_dist = prev_masked_dist = prev_causal_dist = None

        # full_feature: prediction using all state variables
        # masked_feature: prediction using state variables specified by mask
        # causal_feature: prediction using causal parents (inferred so far)
        full_feature = feature if "full" in forward_mode else None
        masked_feature = feature if "masked" in forward_mode else None
        causal_feature = feature if "causal" in forward_mode else None

        if abstraction_mode:
            assert not self.use_cache
            forward_mode = ("causal",)
            full_feature = masked_feature = None
            if self.abstraction_quested:
                if self.continuous_state:
                    causal_feature = causal_feature[:, self.abstraction_idxes]
                else:
                    causal_feature = [causal_feature[idx] for idx in self.abstraction_idxes]

        modes = ["full", "masked", "causal"]
        assert all([ele in modes for ele in forward_mode])
        if "masked" in forward_mode:
            assert mask is not None

        full_dists, masked_dists, causal_dists = [], [], []
        sa_feature_cache = []

        if not self.continuous_action:
            actions = F.one_hot(actions.squeeze(dim=-1), self.action_dim).float()   # (bs, n_pred_step, action_dim)

        actions = torch.unbind(actions, dim=-2)                                     # [(bs, action_dim)] * n_pred_step
        for i, action in enumerate(actions):
            if self.use_cache and self.sa_feature_cache:
                # only used when evaluate with the same state and action a lot in self.update_mask()
                action_feature, full_state_feature = self.sa_feature_cache[i]
            else:
                action_feature, full_state_feature = None, None

            full_dist = masked_dist = None
            if abstraction_mode and self.abstraction_quested:
                causal_dist = self.forward_step_abstraction(causal_feature, action)
            else:
                full_dist, masked_dist, causal_dist = \
                    self.forward_step(full_feature, masked_feature, causal_feature, action, mask,
                                      action_feature, full_state_feature)

            full_dist, full_feature = get_dist_and_feature(prev_full_dist, full_dist)
            masked_dist, masked_feature = get_dist_and_feature(prev_masked_dist, masked_dist)
            causal_dist, causal_feature = get_dist_and_feature(prev_causal_dist, causal_dist)

            prev_full_dist, prev_masked_dist, prev_causal_dist = full_dist, masked_dist, causal_dist

            full_dists.append(full_dist)
            masked_dists.append(masked_dist)
            causal_dists.append(causal_dist)

            sa_feature_cache.append((self.action_feature, self.full_state_feature))

        if self.use_cache and self.sa_feature_cache is None:
            self.sa_feature_cache = sa_feature_cache

        dists = [full_dists, masked_dists, causal_dists]
        result_dists = []
        for mode in forward_mode:
            dist = dists[modes.index(mode)]
            dist = self.stack_dist(dist)
            if reshaped:
                dist = self.restore_batch_size_shape(dist, bs)
            result_dists.append(dist)

        if len(forward_mode) == 1:
            return result_dists[0]

        return result_dists

    def restore_batch_size_shape(self, dist, bs):
        # restore multi-dimensional batch size
        if self.continuous_state:
            mu, std = dist.mean, dist.stddev                                    # (bs, n_pred_step, feature_dim)
            mu = mu.view(*bs, *mu.shape[-2:])                                   # (*bs, n_pred_step, feature_dim)
            std = std.view(*bs, *std.shape[-2:])                                # (*bs, n_pred_step, feature_dim)
            return Normal(mu, std)
        else:
            # [(bs, n_pred_step, feature_i_dim)] * feature_dim
            dist_list = []
            for dist_i in dist:
                if isinstance(dist_i, Normal):
                    mu, std = dist.mean, dist.stddev                            # (bs, n_pred_step, feature_i_dim)
                    mu = mu.view(*bs, *mu.shape[-2:])                           # (*bs, n_pred_step, feature_i_dim)
                    std = std.view(*bs, *std.shape[-2:])                        # (*bs, n_pred_step, feature_i_dim)
                    dist_i = Normal(mu, std)
                elif isinstance(dist_i, OneHotCategorical):
                    logits = dist_i.logits                                      # (bs, n_pred_step, feature_i_dim)
                    logits = logits.view(*bs, *logits.shape[-2:])               # (*bs, n_pred_step, feature_i_dim)
                    dist_i = OneHotCategorical(logits=logits)
                else:
                    raise NotImplementedError
                dist_list.append(dist_i)

            return dist_list

    def forward(self, obs, actions, mask=None, forward_mode=("full", "masked", "causal"),
                abstraction_mode=False):
        feature = self.get_feature(obs)
        return self.forward_with_feature(feature, actions, mask, forward_mode, abstraction_mode)

    def setup_annealing(self, step):
        super(InferenceCMI, self).setup_annealing(step)

    def get_mask_by_id(self, mask_ids):
        """
        :param mask_ids: (bs_1, bs_2, ..., bs_n, feature_dim), idxes of state variable to drop
        :return: (bs_1, bs_2, ..., bs_n, feature_dim, feature_dim + 1), bool mask of state variables to use
        """
        int_mask = F.one_hot(mask_ids, self.feature_dim + 1)
        bool_mask = int_mask < 1
        return bool_mask

    def get_training_mask(self, batch_size):
        # uniformly select one state variable to omit when predicting the next time step value
        idxes = torch.randint(self.feature_dim, (batch_size, self.feature_dim))
        return self.get_mask_by_id(idxes)  # (bs, feature_dim, feature_dim + 1)

    def get_eval_mask(self, batch_size, i):
        # omit i-th state variable or the action when predicting the next time step value
        feature_dim = self.feature_dim

        idxes = torch.full(size=(batch_size, feature_dim), fill_value=i, dtype=torch.int64, device=self.device)
        self_mask = torch.arange(feature_dim, device=self.device)
        # each state variable must depend on itself when predicting the next time step value
        idxes[idxes >= self_mask] += 1

        return self.get_mask_by_id(idxes)  # (bs, feature_dim, feature_dim + 1)

    def prediction_loss_from_multi_dist(self, pred_next_dist, next_feature):
        """
        calculate prediction loss for each prediction distributions
        if use CNN encoder: prediction loss = KL divergence
        else: prediction loss = -log_prob
        :param pred_next_dist:
            a list of prediction distributions under different prediction mode,
            where each element is the next step value for all state variables in the format of distribution as follows,
            if state space is continuous:
                a Normal distribution of shape (bs, n_pred_step, feature_dim)
            else:
                a list of distributions, [OneHotCategorical / Normal] * feature_dim,
                each of shape (bs, n_pred_step, feature_i_dim)
        :param next_feature:
            if use a CNN encoder:
                a Normal distribution of shape (bs, n_pred_step, feature_dim)
            elif state space is continuous:
                a tensor of shape (bs, n_pred_step, feature_dim)
            else:
                a list of tensors, [(bs, n_pred_step, feature_i_dim)] * feature_dim
        :return: prediction loss and {"loss_name": loss_value}
        """
        # (bs, n_pred_step, feature_dim)
        pred_losses = [self.prediction_loss_from_dist(pred_next_dist_i, next_feature)
                       for pred_next_dist_i in pred_next_dist]

        if len(pred_losses) == 2:
            pred_losses.append(None)
        assert len(pred_losses) == 3
        full_pred_loss, masked_pred_loss, causal_pred_loss = pred_losses

        full_pred_loss = full_pred_loss.sum(dim=-1).mean()
        masked_pred_loss = masked_pred_loss.sum(dim=-1).mean()

        pred_loss = full_pred_loss + masked_pred_loss

        pred_loss_detail = {"full_pred_loss": full_pred_loss,
                            "masked_pred_loss": masked_pred_loss}

        if causal_pred_loss is not None:
            causal_pred_loss = causal_pred_loss.sum(dim=-1).mean()
            pred_loss += causal_pred_loss
            pred_loss_detail["causal_pred_loss"] = causal_pred_loss

        return pred_loss, pred_loss_detail

    def update(self, obs, actions, next_obses, eval=False):
        """
        :param obs: {obs_i_key: (bs, obs_i_shape)}
        :param actions: (bs, n_pred_step, action_dim)
        :param next_obses: {obs_i_key: (bs, n_pred_step, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        self.update_num += 1

        eval_freq = self.cmi_params.eval_freq
        inference_gradient_steps = self.params.training_params.inference_gradient_steps
        forward_mode = ("full", "masked", "causal")

        bs = actions.size(0)
        mask = self.get_training_mask(bs)                           # (bs, feature_dim, feature_dim + 1)

        feature = self.encoder(obs)
        next_feature = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(feature, actions, mask, forward_mode=forward_mode)

        # prediction loss in the state / latent space, (bs, n_pred_step)
        if not self.update_num % (eval_freq * inference_gradient_steps):
            pred_next_dist = pred_next_dist[:2]
        pred_loss, loss_detail = self.prediction_loss_from_multi_dist(pred_next_dist, next_feature)

        loss = pred_loss

        if not eval and torch.isfinite(loss):
            self.backprop(loss, loss_detail)

        return loss_detail

    def update_mask(self, obs, actions, next_obses):
        bs = actions.size(0)
        feature_dim = self.feature_dim

        # set up cache for faster computation
        self.use_cache = True
        self.sa_feature_cache = None

        eval_details = {}

        masked_pred_losses = []
        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder(next_obses)

            for i in range(feature_dim):
                mask = self.get_eval_mask(bs, i)
                if i == 0:
                    pred_next_dists = self.forward_with_feature(feature, actions, mask)
                    # pred_loss: (bs, n_pred_step, feature_dim)
                    full_pred_loss, masked_pred_loss, eval_pred_loss = \
                        [self.prediction_loss_from_dist(pred_next_dist_i, next_feature, keep_variable_dim=True)
                         for pred_next_dist_i in pred_next_dists]
                else:
                    pred_next_dist = self.forward_with_feature(feature, actions, mask, forward_mode=("masked",))
                    # pred_loss: (bs, n_pred_step, feature_dim)
                    masked_pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_feature,
                                                                      keep_variable_dim=True)

                masked_pred_loss = masked_pred_loss.mean(dim=1)                         # (bs, feature_dim)
                masked_pred_losses.append(masked_pred_loss)
            full_pred_loss = full_pred_loss.mean(dim=1)[..., None]                      # (bs, feature_dim, 1)
            eval_pred_loss = eval_pred_loss.sum(dim=(1, 2)).mean()                      # scalar
            eval_details["eval_pred_loss"] = eval_pred_loss

        masked_pred_losses = torch.stack(masked_pred_losses, dim=-1)                    # (bs, feature_dim, feature_dim)

        # clean cache
        self.use_cache = False
        self.sa_feature_cache = None
        self.action_feature = None
        self.full_state_feature = None

        # full_pred_loss uses all state variables + action,
        # while along dim 1 of, masked_pred_losses drops either one state variable or the action
        CMI = masked_pred_losses - full_pred_loss                                       # (bs, feature_dim, feature_dim)
        CMI = CMI.mean(dim=0)                                                           # (feature_dim, feature_dim)

        self.eval_step_CMI += CMI
        self.mask_update_idx += 1

        eval_steps = self.cmi_params.eval_steps
        eval_tau = self.cmi_params.eval_tau
        if self.mask_update_idx == eval_steps:
            self.eval_step_CMI /= eval_steps

            eval_step_CMI = torch.eye(feature_dim, feature_dim + 1, dtype=torch.float32, device=self.device)
            eval_step_CMI *= self.CMI_threshold

            # (feature_dim, feature_dim), (feature_dim, feature_dim)
            upper_tri, lower_tri = torch.triu(self.eval_step_CMI), torch.tril(self.eval_step_CMI, diagonal=-1)
            eval_step_CMI[:, 1:] += upper_tri
            eval_step_CMI[:, :-1] += lower_tri

            self.mask_CMI = self.mask_CMI * eval_tau + eval_step_CMI * (1 - eval_tau)
            self.mask = self.mask_CMI >= self.CMI_threshold
            self.mask[self.diag_mask] = True

        return eval_details

    def reward(self, obs, actions, next_obses, output_numpy=False):
        """
        Calculate reward for RL policy
        :param obs: (bs, obs_spec) during policy training or (obs_spec,) after env.step()
        :param actions: (bs, n_pred_step, action_dim) during policy training or (action_dim,) after env.step()
        :param next_obses: (bs, n_pred_step, obs_spec) during policy training or (obs_spec,) after env.step()
        :param output_numpy: output numpy or tensor
        :return: (bs, n_pred_step, 1) or scalar
        """
        obs, actions, next_obses, reward_need_squeeze = self.preprocess(obs, actions, next_obses)

        with torch.no_grad():
            full_next_dist, causal_next_dist = self.forward(obs, actions, forward_mode=("full", "causal",))
            next_features = self.encoder(next_obses)
            full_neg_log_prob = self.prediction_loss_from_dist(full_next_dist, next_features)       # (bs, n_pred_step)
            causal_neg_log_prob = self.prediction_loss_from_dist(causal_next_dist, next_features)   # (bs, n_pred_step)

            causal_pred_reward = full_neg_log_prob

            normalized_causal_pred_reward = torch.tanh((causal_pred_reward - self.causal_pred_reward_mean) /
                                                       (self.causal_pred_reward_std * 2))

            tau = 0.99
            if len(causal_pred_reward) > 0:
                batch_mean = causal_pred_reward.mean(dim=0)
                batch_std = causal_pred_reward.std(dim=0, unbiased=False)
                self.causal_pred_reward_mean = self.causal_pred_reward_mean * tau + batch_mean * (1 - tau)
                self.causal_pred_reward_std = self.causal_pred_reward_std * tau + batch_std * (1 - tau)

            pred_diff_reward = causal_neg_log_prob - full_neg_log_prob                  # (bs, n_pred_step)
            normalized_pred_diff_reward = torch.tanh(pred_diff_reward / (self.pred_diff_reward_std * 2))
            if len(pred_diff_reward) > 0:
                batch_std = pred_diff_reward.std(dim=0, unbiased=False)
                self.pred_diff_reward_std = self.pred_diff_reward_std * tau + batch_std * (1 - tau)

            causal_pred_reward_weight = self.cmi_params.causal_pred_reward_weight
            pred_diff_reward_weight = self.cmi_params.pred_diff_reward_weight
            reward = causal_pred_reward_weight * normalized_causal_pred_reward + \
                     pred_diff_reward_weight * normalized_pred_diff_reward

            reward = reward[..., None]                                          # (bs, n_pred_step, 1)

        reward = self.reward_postprocess(reward, reward_need_squeeze, output_numpy)

        return reward

    def eval_prediction(self, obs, actions, next_obses):
        obs, actions, next_obses, _ = self.preprocess(obs, actions, next_obses)

        with torch.no_grad():
            feature = self.encoder(obs)
            next_feature = self.encoder(next_obses)
            pred_next_dist = self.forward_with_feature(feature, actions, forward_mode=("causal",))
            pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_feature)

            accuracy = None
            if not self.continuous_state:
                accuracy = []
                for dist_i, next_feature_i in zip(pred_next_dist, next_feature):
                    if not isinstance(dist_i, OneHotCategorical):
                        continue
                    logits = dist_i.logits                                 # (bs, n_pred_step, feature_i_inner_dim)
                    # (bs, n_pred_step)
                    accuracy_i = logits.argmax(dim=-1) == next_feature_i.argmax(dim=-1)
                    accuracy.append(accuracy_i)
                accuracy = torch.stack(accuracy, dim=-1)
                accuracy = to_numpy(accuracy)

        return pred_next_dist, next_feature, pred_loss, accuracy

    def get_mask(self):
        return self.mask

    def get_state_abstraction(self):
        self.abstraction_quested = True
        abstraction_graph = self.update_abstraction()
        self.update_abstracted_dynamics()
        return abstraction_graph

    def update_abstraction(self):
        self.abstraction_graph = get_state_abstraction(to_numpy(self.get_mask()))
        self.abstraction_idxes = list(self.abstraction_graph.keys())

        action_idx = self.feature_dim
        self.action_children_idxes = [idx for idx, parent_idxes in self.abstraction_graph.items()
                                      if action_idx in parent_idxes]
        self.abstraction_adjacency = {}
        for idx, parents in self.abstraction_graph.items():
            self.abstraction_adjacency[idx] = [parent for parent in parents if parent < action_idx]

        return self.abstraction_graph

    def update_abstracted_dynamics(self,):
        # only need to calculate action feature for state variables that are children of the action
        action_children_idxes = self.action_children_idxes
        self.abstraction_action_feature_weights = [w[action_children_idxes]
                                                   for w in self.action_feature_weights]
        self.abstraction_action_feature_biases = [b[action_children_idxes]
                                                  for b in self.action_feature_biases]

        # when predicting each state variables in the abstraction, only need to compute state feature for their parents
        feature_dim = self.feature_dim
        self.abstraction_state_feature_1st_layer_weights = {}
        self.abstraction_state_feature_1st_layer_biases = {}
        idxes = []
        for idx, parent_idxes in self.abstraction_adjacency.items():
            idxes.extend([parent_idx + idx * feature_dim for parent_idx in parent_idxes])
            self.abstraction_state_feature_1st_layer_weights[idx] = \
                [w[idx:idx + 1] for i, w in enumerate(self.state_feature_1st_layer_weights) if i in parent_idxes]
            self.abstraction_state_feature_1st_layer_biases[idx] = \
                [b[idx:idx + 1] for i, b in enumerate(self.state_feature_1st_layer_biases) if i in parent_idxes]

        self.abstraction_state_feature_weights = [w[idxes] for w in self.state_feature_weights]
        self.abstraction_state_feature_biases = [b[idxes] for b in self.state_feature_biases]

        abstraction_idxes = self.abstraction_idxes
        self.abstraction_generative_weights = [w[abstraction_idxes] for w in self.generative_weights]
        self.abstraction_generative_biases = [b[abstraction_idxes] for b in self.generative_biases]
        self.abstraction_generative_last_layer_weights = \
            [w for i, w in enumerate(self.generative_last_layer_weights) if i in abstraction_idxes]
        self.abstraction_generative_last_layer_biases = \
            [b for i, b in enumerate(self.generative_last_layer_biases) if i in abstraction_idxes]

    def get_adjacency(self):
        return self.mask_CMI[:, :-1]

    def get_intervention_mask(self):
        return self.mask_CMI[:, -1:]

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(training=False)

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask_CMI": self.mask_CMI,
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("inference loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.mask_CMI = checkpoint["mask_CMI"]
            self.mask = self.mask_CMI >= self.CMI_threshold
            self.mask_CMI[self.diag_mask] = self.CMI_threshold
            self.mask[self.diag_mask] = True
