import numpy as np
from collections import deque, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gumbel import gumbel_sigmoid


def reset_layer(w, b):
    nn.init.kaiming_uniform_(w, nonlinearity='relu')
    fan_in = w.shape[1]
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(b, -bound, bound)


def reparameterize(mu, log_std):
    std = torch.exp(log_std)
    eps = torch.randn_like(std)
    return eps * std + mu


def forward_network(input, weights, biases, activation=F.relu):
    """
    given an input and a multi-layer networks (i.e., a list of weights and a list of biases),
        apply the network to each input, and return output
    the same activation function is applied to all layers except for the last layer
    """
    x = input
    for i, (w, b) in enumerate(zip(weights, biases)):
        # x (p_bs, bs, in_dim), bs: data batch size which must be 1D
        # w (p_bs, in_dim, out_dim), p_bs: parameter batch size
        # b (p_bs, 1, out_dim)
        x = torch.bmm(x, w) + b
        if i < len(weights) - 1 and activation:
            x = activation(x)
    return x


def forward_network_batch(inputs, weights, biases, activation=F.relu):
    """
    given a list of inputs and a list of ONE-LAYER networks (i.e., a list of weights and a list of biases),
        apply each network to each input, and return a list
    """
    x = []
    for x_i, w, b in zip(inputs, weights, biases):
        # x_i (p_bs, bs, in_dim), bs: data batch size which must be 1D
        # w (p_bs, in_dim, out_dim), p_bs: parameter batch size
        # b (p_bs, 1, out_dim)
        x_i = torch.bmm(x_i, w) + b
        if activation:
            x_i = activation(x_i)
        x.append(x_i)
    return x


def forward_gated_network(input, weights, biases, gate_weights, gate_biases, deterministic=False, activation=F.relu):
    """
    given an input and a multi-layer networks (i.e., a list of weights and a list of biases),
        apply the network to each input, and return output
    the same activation function is applied to all layers except for the last layer
    """
    gate = None
    if len(gate_weights):
        gate_log_alpha = input
        for i, (w, b) in enumerate(zip(gate_weights, gate_biases)):
            # gate_log_alpha (bs, p_bs, in_dim), bs: data batch size
            # w (p_bs, out_dim, in_dim), p_bs: parameter batch size
            # b (p_bs, out_dim)
            gate_log_alpha = gate_log_alpha.unsqueeze(dim=-2)         # (bs, p_bs, 1, in_dim)
            gate_log_alpha = (gate_log_alpha * w).sum(dim=-1) + b     # (bs, p_bs, out_dim)
            if i < len(gate_weights) - 1 and activation:
                gate_log_alpha = activation(gate_log_alpha)

        if deterministic:
            gate = (gate_log_alpha > 0).float()
        else:
            gate = gumbel_sigmoid(gate_log_alpha, device=gate_log_alpha.device, hard=True)

    x = input
    for i, (w, b) in enumerate(zip(weights, biases)):
        # x (bs, p_bs, in_dim), bs: data batch size
        # w (p_bs, out_dim, in_dim), p_bs: parameter batch size
        # b (p_bs, out_dim)
        x = x.unsqueeze(dim=-2)                         # (bs, p_bs, 1, in_dim)
        x = (x * w).sum(dim=-1) + b                     # (bs, p_bs, out_dim)
        if i < len(weights) - 1 and activation:
            x = activation(x)
        if i == len(weights) - 2 and gate is not None:
            x = x * gate
    return x


def get_controllable(mask):
    feature_dim = mask.shape[0]
    M = mask[:, :feature_dim]
    I = mask[:, feature_dim:]

    # feature that are directly affected by actions
    action_children = []
    for i in range(feature_dim):
        if I[i].any():
            action_children.append(i)

    # decedents of those features
    controllable = []
    queue = deque(action_children)
    while len(queue):
        feature_idx = queue.popleft()
        controllable.append(feature_idx)
        for i in range(feature_dim):
            if M[i, feature_idx] and (i not in controllable) and (i not in queue):
                queue.append(i)
    return controllable


def get_state_abstraction(mask):
    feature_dim = mask.shape[0]
    M = mask[:, :feature_dim]

    controllable = get_controllable(mask)
    # ancestors of controllable features
    action_relevant = []
    queue = deque(controllable)
    while len(queue):
        feature_idx = queue.popleft()
        if feature_idx not in controllable:
            action_relevant.append(feature_idx)
        for i in range(feature_dim):
            if (i not in controllable + action_relevant) and (i not in queue):
                if M[feature_idx, i]:
                    queue.append(i)

    abstraction_idx = list(set(controllable + action_relevant))
    abstraction_idx.sort()

    abstraction_graph = OrderedDict()
    for idx in abstraction_idx:
        abstraction_graph[idx] = [i for i, e in enumerate(mask[idx]) if e]

    return abstraction_graph
