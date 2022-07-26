import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical


from model.inference import Inference
from model.inference_utils import reset_layer, forward_network, forward_network_batch


class Attention(nn.Module):
    def __init__(self, num_query, query_dim, num_key, key_dim, attention_dim):
        super(Attention, self).__init__()
        self.temperature = a = np.sqrt(attention_dim)
        self.key_weight = nn.Parameter(torch.FloatTensor(num_key, key_dim, attention_dim).uniform_(-a, a))
        self.key_bias = nn.Parameter(torch.zeros(num_key, 1, attention_dim))
        self.query_weight = nn.Parameter(torch.FloatTensor(num_query, query_dim, attention_dim).uniform_(-a, a))
        self.query_bias = nn.Parameter(torch.zeros(num_query, 1, attention_dim))

    def forward(self, q, k):
        """
        :param q: (num_query, bs, query_dim)
        :param k: (num_key, bs, key_dim)
        :return:
        """
        query = torch.bmm(q, self.query_weight) + self.query_bias       # (num_query, bs, attention_dim)
        key = torch.bmm(k, self.key_weight) + self.key_bias             # (num_key, bs, attention_dim)

        query = query.permute(1, 0, 2)                                  # (bs, num_query, attention_dim)
        key = key.permute(1, 2, 0)                                      # (bs, attention_dim, num_key)

        return torch.bmm(query, key) / self.temperature                 # (bs, num_query, num_key)


class InferenceNPS(Inference):
    def __init__(self, encoder, params):
        self.nps_params = params.inference_params.nps_params
        super(InferenceNPS, self).__init__(encoder, params)

    def init_model(self):
        params = self.params
        nps_params = self.nps_params

        # model params
        self.continuous_state = continuous_state = params.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = self.encoder.feature_inner_dim
        self.num_rule = num_rule = nps_params.num_rule
        self.rule_embed_dim = rule_embed_dim = nps_params.rule_embed_dim

        self.action_feature_weights = nn.ParameterList()
        self.action_feature_biases = nn.ParameterList()
        self.state_feature_weights = nn.ParameterList()
        self.state_feature_biases = nn.ParameterList()
        self.rule_weights = nn.ParameterList()
        self.rule_biases = nn.ParameterList()

        self.rule_embed = nn.Parameter(torch.randn(num_rule, 1, rule_embed_dim, dtype=torch.float32))
        feature_embed_dim = nps_params.feature_fc_dims[-1]
        self.rule_selector = Attention(feature_dim, feature_embed_dim, num_rule, rule_embed_dim,
                                       nps_params.rule_selector_dim)
        self.cond_selector = Attention(feature_dim, feature_embed_dim, feature_dim + 1, feature_embed_dim,
                                       nps_params.cond_selector_dim)

        # only needed for discrete state space
        self.state_feature_1st_layer_weights = nn.ParameterList()
        self.state_feature_1st_layer_biases = nn.ParameterList()
        self.rule_last_layer_weights = nn.ParameterList()
        self.rule_last_layer_biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable
        # action feature extractor
        in_dim = action_dim
        for out_dim in nps_params.feature_fc_dims:
            self.action_feature_weights.append(nn.Parameter(torch.zeros(1, in_dim, out_dim)))
            self.action_feature_biases.append(nn.Parameter(torch.zeros(1, 1, out_dim)))
            in_dim = out_dim

        # state feature extractor
        if continuous_state:
            in_dim = 1
            fc_dims = nps_params.feature_fc_dims
        else:
            out_dim = nps_params.feature_fc_dims[0]
            fc_dims = nps_params.feature_fc_dims[1:]
            for feature_i_dim in self.feature_inner_dim:
                in_dim = feature_i_dim
                self.state_feature_1st_layer_weights.append(nn.Parameter(torch.zeros(1, in_dim, out_dim)))
                self.state_feature_1st_layer_biases.append(nn.Parameter(torch.zeros(1, 1, out_dim)))
            in_dim = out_dim

        for out_dim in fc_dims:
            self.state_feature_weights.append(nn.Parameter(torch.zeros(feature_dim, in_dim, out_dim)))
            self.state_feature_biases.append(nn.Parameter(torch.zeros(feature_dim, 1, out_dim)))
            in_dim = out_dim

        # rule mlps
        in_dim = feature_embed_dim * 2
        for out_dim in nps_params.rule_fc_dims:
            self.rule_weights.append(nn.Parameter(torch.zeros(num_rule, in_dim, out_dim)))
            self.rule_biases.append(nn.Parameter(torch.zeros(num_rule, 1, out_dim)))
            in_dim = out_dim

        if continuous_state:
            self.rule_weights.append(nn.Parameter(torch.zeros(num_rule, in_dim, 2)))
            self.rule_biases.append(nn.Parameter(torch.zeros(num_rule, 1, 2)))
        else:
            for feature_i_dim in self.feature_inner_dim:
                final_dim = 2 if feature_i_dim == 1 else feature_i_dim
                self.rule_last_layer_weights.append(nn.Parameter(torch.zeros(num_rule, in_dim, final_dim)))
                self.rule_last_layer_biases.append(nn.Parameter(torch.zeros(num_rule, 1, final_dim)))

    def reset_params(self):
        feature_dim = self.feature_dim
        num_rule = self.num_rule
        for w, b in zip(self.action_feature_weights, self.action_feature_biases):
            reset_layer(w, b)
        for w, b in zip(self.state_feature_1st_layer_weights, self.state_feature_1st_layer_biases):
            reset_layer(w, b)
        for w, b in zip(self.state_feature_weights, self.state_feature_biases):
            for i in range(feature_dim):
                reset_layer(w[i], b[i])
        for w, b in zip(self.rule_weights, self.rule_biases):
            for i in range(num_rule):
                reset_layer(w[i], b[i])
        for w, b in zip(self.rule_last_layer_weights, self.rule_last_layer_biases):
            for i in range(num_rule):
                reset_layer(w[i], b[i])

    def forward_step(self, feature, action):
        """
        :param feature:
            if state space is continuous: (bs, feature_dim).
            else: [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        bs = action.shape[0]
        feature_dim = self.feature_dim
        num_rule = self.num_rule
        action = action.unsqueeze(dim=0)                                    # (1, bs, action_dim)
        action_feature = forward_network(action,
                                         self.action_feature_weights,
                                         self.action_feature_biases)        # (1, bs, feature_embed_dim)

        if self.continuous_state:
            x = feature.transpose(0, 1).unsqueeze(dim=-1)                   # (feature_dim, bs, 1)
        else:
            # [(1, bs, feature_i_dim)] * feature_dim
            x = [feature_i.unsqueeze(dim=0) for feature_i in feature]
            x = forward_network_batch(x,
                                      self.state_feature_1st_layer_weights,
                                      self.state_feature_1st_layer_biases)  # [(1, bs, layer_out_dim)] * feature_dim
            x = torch.cat(x, dim=0)                                         # (feature_dim, bs, layer_out_dim)

        state_feature = forward_network(x,
                                        self.state_feature_weights,
                                        self.state_feature_biases)          # (feature_dim, bs, feature_embed_dim)
        sa_feature = torch.cat([action_feature, state_feature], dim=0)      # (feature_dim + 1, bs, feature_embed_dim)

        rule_embed = self.rule_embed.repeat(1, bs, 1)                       # (num_rule, bs, rule_embed_dim)
        rule_logits = self.rule_selector(state_feature, rule_embed)         # (bs, feature_dim, num_rule)
        cond_logits = self.cond_selector(state_feature, sa_feature)         # (bs, feature_dim, feature_dim + 1)
        if self.training:
            rule_select = F.gumbel_softmax(rule_logits, hard=True, dim=-1)  # (bs, feature_dim, num_rule)
            cond_select = F.gumbel_softmax(cond_logits, hard=True, dim=-1)  # (bs, feature_dim, feature_dim + 1)
        else:
            # (bs, feature_dim, num_rule)
            rule_select = F.one_hot(torch.argmax(rule_logits, dim=-1), rule_logits.size(-1)).float()
            # (bs, feature_dim, feature_dim + 1)
            cond_select = F.one_hot(torch.argmax(cond_logits, dim=-1), cond_logits.size(-1)).float()

        cond_select = cond_select.permute(1, 2, 0)                          # (feature_dim, feature_dim + 1, bs)
        cond_select = cond_select.unsqueeze(dim=-1)                         # (feature_dim, feature_dim + 1, bs, 1)
        cond_sa_feature = (sa_feature * cond_select).sum(dim=1)             # (feature_dim, bs, feature_embed_dim)

        rule_input = torch.cat([state_feature, cond_sa_feature], dim=-1)    # (feature_dim, bs, 2 * feature_embed_dim)

        # (num_rule, feature_dim, bs, 2 * feature_embed_dim)
        rule_input = rule_input.repeat(num_rule, 1, 1, 1)

        # (num_rule, bs * feature_dim, 2 * feature_embed_dim)
        rule_input = rule_input.view(num_rule, bs * feature_dim, -1)

        # (num_rule, bs * feature_dim, out_dim)
        x = forward_network(rule_input, self.rule_weights, self.rule_biases)
        x = x.view(num_rule, feature_dim, bs, -1)                           # (num_rule, feature_dim, bs, out_dim)

        rule_select = rule_select.permute(2, 1, 0)                          # (num_rule, feature_dim, bs)
        rule_select = rule_select.unsqueeze(dim=-1)                         # (num_rule, feature_dim, bs, 1)

        def normal_helper(mean_, base_, log_std_):
            if self.residual:
                mean_ = mean_ + base_
            log_std_ = torch.clip(log_std_, min=self.log_std_min, max=self.log_std_max)
            std_ = torch.exp(log_std_)
            return Normal(mean_, std_)

        if self.continuous_state:
            x = (rule_select * x).sum(dim=0)                                # (feature_dim, bs, out_dim)
            x = x.permute(1, 0, 2)                                          # (bs, feature_dim, 2)
            mu, log_std = x.unbind(dim=-1)                                  # (bs, feature_dim) * 2
            return normal_helper(mu, feature, log_std)
        else:
            x = F.relu(x)                                                   # (num_rule, feature_dim, bs, out_dim)
            x = torch.unbind(x, dim=1)                                      # [(num_rule, bs, out_dim)] * feature_dim
            # [(num_rule, bs, feature_i_inner_dim)] * feature_dim
            x = forward_network_batch(x,
                                      self.rule_last_layer_weights,
                                      self.rule_last_layer_biases,
                                      activation=None)

            feature_inner_dim = self.feature_inner_dim
            rule_select = rule_select.unbind(dim=1)                         # [(num_rule, bs, 1)] * feature_dim

            dist = []
            for base_i, feature_i_inner_dim, dist_i, rule_select_i in zip(feature, feature_inner_dim, x, rule_select):
                dist_i = (dist_i * rule_select_i).sum(dim=0)                # (bs, feature_i_inner_dim)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)            # (bs, 1), (bs, 1)
                    dist.append(normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist

    def forward_step_abstraction(self, abstraction_feature, action):
        """
        :param abstraction_feature: (bs, abstraction_feature_dim) or [(bs, feature_i_dim)] * abstraction_feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all abstraction state variables in the format of distribution,
            if state space is continuous, distribution is a tuple for (sample + mean + log_std),
            otherwise, distribution is a tuple for (sample + logits),
        """
        return self.forward_step(abstraction_feature, action)

    def get_state_abstraction(self):
        abstraction_graph = {i: np.arange(self.feature_dim + 1) for i in range(self.feature_dim)}
        return abstraction_graph

    def get_adjacency(self):
        return torch.ones(self.feature_dim, self.feature_dim)

    def get_intervention_mask(self):
        return torch.ones(self.feature_dim, 1)
