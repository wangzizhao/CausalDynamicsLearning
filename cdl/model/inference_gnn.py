import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.inference_mlp import InferenceMLP
from model.inference_utils import reparameterize


class InferenceGNN(InferenceMLP):
    def __init__(self, encoder, params):
        self.gnn_params = params.inference_params.gnn_params
        super(InferenceGNN, self).__init__(encoder, params)

    def init_model(self):
        params = self.params
        device = self.device
        gnn_params = self.gnn_params

        # model params
        self.continuous_state = continuous_state = params.continuous_state

        self.action_dim = action_dim = params.action_dim
        self.feature_dim = feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = feature_inner_dim = self.encoder.feature_inner_dim

        self.node_attr_dim = node_attr_dim = gnn_params.node_attr_dim
        self.edge_attr_dim = edge_attr_dim = gnn_params.edge_attr_dim

        self.embedders = nn.ModuleList()
        for i in range(feature_dim):
            in_dim = 1 if continuous_state else feature_inner_dim[i]
            embedder_i = []
            for out_dim in gnn_params.embedder_dims:
                embedder_i.append(nn.Linear(in_dim, out_dim))
                embedder_i.append(nn.ReLU())
                in_dim = out_dim
            embedder_i.append(nn.Linear(in_dim, node_attr_dim))
            embedder_i = nn.Sequential(*embedder_i)
            self.embedders.append(embedder_i)

        edge_net = []
        in_dim = node_attr_dim * 2
        for out_dim in gnn_params.edge_net_dims:
            edge_net.append(nn.Linear(in_dim, out_dim))
            edge_net.append(nn.ReLU())
            in_dim = out_dim
        edge_net.append(nn.Linear(in_dim, edge_attr_dim))
        self.edge_net = nn.Sequential(*edge_net)

        node_net = []
        in_dim = node_attr_dim + edge_attr_dim + action_dim
        for out_dim in gnn_params.node_net_dims:
            node_net.append(nn.Linear(in_dim, out_dim))
            node_net.append(nn.ReLU())
            in_dim = out_dim
        node_net.append(nn.Linear(in_dim, node_attr_dim))
        self.node_net = nn.Sequential(*node_net)

        self.projectors = nn.ModuleList()
        for i in range(feature_dim):
            in_dim = node_attr_dim
            final_dim = 2 if continuous_state or feature_inner_dim[i] == 1 else feature_inner_dim[i]
            projector_i = []
            for out_dim in gnn_params.projector_dims:
                projector_i.append(nn.Linear(in_dim, out_dim))
                projector_i.append(nn.ReLU())
                in_dim = out_dim
            projector_i.append(nn.Linear(in_dim, final_dim))
            projector_i = nn.Sequential(*projector_i)
            self.projectors.append(projector_i)

        adj_full = torch.ones(feature_dim, feature_dim, device=device) - torch.eye(feature_dim, device=device)
        edge_pair = adj_full.nonzero(as_tuple=False)
        self.edge_left_idxes = edge_pair[:, 0]
        self.edge_right_idxes = edge_pair[:, 1]

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
        if self.continuous_state:
            feature_ = torch.split(feature, 1, dim=-1)                          # [(bs, feature_i_dim)] * feature_dim
        else:
            feature_ = feature

        embeddings = []
        for feature_i, embedder_i in zip(feature_, self.embedders):
            embeddings.append(embedder_i(feature_i.float()))                    # [(bs, node_attr_dim)] * feature_dim
        embeddings = torch.stack(embeddings, dim=-2)                            # (bs, feature_dim, node_attr_dim)

        # get edge attr, num_edge = feature_dim * (feature_dim - 1)
        edge_left = embeddings[..., self.edge_left_idxes, :]                    # (bs, num_edge, node_attr_dim)
        edge_right = embeddings[..., self.edge_right_idxes, :]                  # (bs, num_edge, node_attr_dim)
        edge_input = torch.cat([edge_left, edge_right], dim=-1)                 # (bs, num_edge, node_attr_dim * 2)
        edge_attr = self.edge_net(edge_input)                                   # (bs, num_edge, edge_attr_dim)

        bs = edge_attr.shape[:-2]
        feature_dim = self.feature_dim
        # (bs, feature_dim, feature_dim - 1, edge_attr_dim)
        edge_attr = edge_attr.reshape(*bs, feature_dim, feature_dim - 1, self.edge_attr_dim)
        edge_attr = edge_attr.sum(dim=-2)                                       # (bs, feature_dim, edge_attr_dim)

        action = action.unsqueeze(-2)                                           # (bs, 1, action_dim)
        action = action.expand(*bs, feature_dim, -1)                            # (bs, feature_dim, action_dim)

        # (bs, feature_dim, node_attr_dim + action_dim + edge_attr_dim)
        node_input = torch.cat([embeddings, action, edge_attr], dim=-1)
        next_node_attr = self.node_net(node_input)                              # (bs, feature_dim, node_attr_dim)

        next_features = []
        next_node_attr = torch.unbind(next_node_attr, dim=-2)                   # [(bs, node_attr_dim)] * feature_dim
        for next_node_attr_i, projector_i in zip(next_node_attr, self.projectors):
            next_features.append(projector_i(next_node_attr_i))                 # [(bs, feature_i_dim)] * feature_dim

        if self.continuous_state:
            next_features = torch.stack(next_features, dim=-2)                  # (bs, feature_dim, 2)
            mu, log_std = torch.unbind(next_features, dim=-1)                   # (bs, feature_dim), (bs, feature_dim)
            return self.normal_helper(mu, feature, log_std)
        else:
            dist = []
            for feature_i, feature_i_inner_dim, dist_i in zip(feature, self.feature_inner_dim, next_features):
                if feature_i_inner_dim == 1:
                    mu, log_std = torch.split(dist_i, 1, dim=-1)            # (bs, 1), (bs, 1)
                    dist.append(self.normal_helper(mu, base_i, log_std))
                else:
                    dist.append(OneHotCategorical(logits=dist_i))
            return dist
