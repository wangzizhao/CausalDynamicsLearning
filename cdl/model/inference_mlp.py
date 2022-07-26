import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference import Inference
from model.inference_utils import reparameterize


class InferenceMLP(Inference):
    def __init__(self, encoder, params):
        self.mlp_params = params.inference_params.mlp_params
        super(InferenceMLP, self).__init__(encoder, params)

    def init_model(self):
        params = self.params
        mlp_params = self.mlp_params

        # model params
        self.continuous_state = continuous_state = params.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = feature_inner_dim = self.encoder.feature_inner_dim

        if continuous_state:
            in_dim = feature_dim + action_dim
            final_dim = 2 * feature_dim
        else:
            in_dim = np.sum(feature_inner_dim) + action_dim
            # if feature_i_inner_dim = 1, it means it is a continuous state variable, so need to output its mean and std
            final_dim = np.sum(feature_inner_dim) + np.sum(feature_inner_dim == 1)

        fcs = []
        for out_dim in mlp_params.fc_dims:
            fcs.append(nn.Linear(in_dim, out_dim))
            fcs.append(nn.ReLU())
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, final_dim))

        self.fcs = nn.Sequential(*fcs)

    def forward_step(self, feature, action):
        """
        :param feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param action: (bs, action_dim)
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous, distribution is a tuple for (sample, mean, log_std), where each element is
            of shape (bs, feature_dim)
            otherwise, distribution is a tuple for (sample, logits), where each element is of shape
            [(bs, feature_i_dim)] * feature_dim
        """

        if not self.continuous_state:
            feature_copy = feature
            feature = torch.cat(feature, dim=-1)

        inputs = torch.cat([feature, action], dim=-1)
        dist = self.fcs(inputs)

        if self.continuous_state:
            mu, log_std = torch.split(dist, int(self.feature_dim), dim=-1)              # (bs, feature_dim) * 2
            return self.normal_helper(mu, feature, log_std)
        else:
            split_sections = [2 if feature_i_inner_dim == 1 else feature_i_inner_dim
                              for feature_i_inner_dim in self.feature_inner_dim]
            raw_dist = torch.split(dist, split_sections, dim=-1)

            dist = []
            for base_i, feature_i_inner_dim, dist_i in zip(feature_copy, self.feature_inner_dim, raw_dist):
                dist_i = dist_i.squeeze(dim=0)
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)            # (bs, 1), (bs, 1)
                    dist.append(self.normal_helper(mu, base_i, log_std))
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
