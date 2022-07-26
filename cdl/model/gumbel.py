import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

EPS = 1e-6


def sample_logistic(shape, device):
    u = torch.rand(shape, dtype=torch.float32, device=device)
    u = torch.clip(u, EPS, 1 - EPS)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, device, bs=None, tau=1, hard=False):
    if bs is None:
        shape = log_alpha.shape
    else:
        shape = bs + log_alpha.shape

    logistic_noise = sample_logistic(shape, device)
    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


class GumbelMatrix(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    """
    def __init__(self, shape, init_value, flip_prob, device):
        super(GumbelMatrix, self).__init__()
        self.device = device
        self.flip_prob = flip_prob
        self.shape = shape
        self.log_alpha = torch.nn.Parameter(torch.zeros(shape))
        self.reset_parameters(init_value)

    def forward(self, bs, tau=1, drawhard=True):
        if self.training:
            sample = gumbel_sigmoid(self.log_alpha, self.device, bs, tau=tau, hard=drawhard)
        else:
            sample = (self.log_alpha > 0).float()
        return sample

    def get_prob(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)

    def reset_parameters(self, init_value):
        log_alpha_init = -np.log(1 / init_value - 1)
        torch.nn.init.constant_(self.log_alpha, log_alpha_init)


class ConditionalGumbelMatrix(torch.nn.Module):
    """
    Random matrix M used for the mask that's conditioned on state and action.
    Can sample a matrix and backpropagate using the Gumbel straigth-through estimator.
    """
    def __init__(self, feature_dim, action_dim, final_dim, fc_dims, flip_prob, device):
        super(ConditionalGumbelMatrix, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.final_dim = final_dim
        self.flip_prob = flip_prob
        self.device = device
        self.fc_dims = fc_dims
        self.update_uniform(0, 1)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        # Instantiate the parameters of each layer in the model of each variable
        in_dim = feature_dim + action_dim
        for out_dim in fc_dims + [final_dim]:
            self.weights.append(nn.Parameter(torch.zeros(feature_dim, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(feature_dim, out_dim)))
            in_dim = out_dim
        self.reset_params()

        self.causal_feature_idxes = None
        self.causal_weights = None
        self.causal_biases = None

    def reset_params(self):
        in_dim = self.feature_dim + self.action_dim
        for w, b, fan_in in zip(self.weights, self.biases, [in_dim] + self.fc_dims):
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    def forward_fcs(self, feature, action, forward_idxes=None):
        """
        :param feature: (bs, feature_dim)
        :param action: (bs, action_dim)
        """
        out = torch.cat([feature, action], dim=-1)                  # (bs, num_forward_idxes, feature_dim + action_dim)

        weights = self.causal_weights if forward_idxes else self.weights
        biases = self.causal_biases if forward_idxes else self.biases
        for i, (w, b) in enumerate(zip(weights, biases)):
            out = out[:, :, None]                                   # (bs, num_forward_idxes, 1, in_dim)
            out = torch.sum(w * out, dim=-1) + b                    # (bs, num_forward_idxes, out_dim)
            if i < len(weights) - 1:
                out = F.leaky_relu(out)

        return out

    def forward(self, feature, action, forward_idxes, tau=1, drawhard=True):
        log_alpha = self.forward_fcs(feature, action, forward_idxes)
        if self.training:
            sample = gumbel_sigmoid(log_alpha, self.uniform, self.device, bs=None, tau=tau, hard=drawhard)
        else:
            sample = torch.sigmoid(log_alpha)
        prob = torch.sigmoid(log_alpha)
        return sample, prob

    def setup_causal_feature_idxes(self, causal_feature_idxes):
        self.causal_feature_idxes = causal_feature_idxes
        self.causal_weights = [w[causal_feature_idxes] for w in self.weights]
        self.causal_biases = [b[causal_feature_idxes] for b in self.biases]

    def get_prob(self, feature, action):
        """Returns probability of getting one"""
        log_alpha = self.forward_fcs(feature, action)
        return torch.sigmoid(log_alpha)

    def update_uniform(self, low, high):
        low = torch.tensor(low, dtype=torch.float32, device=self.device)
        high = torch.tensor(high, dtype=torch.float32, device=self.device)
        self.uniform = torch.distributions.uniform.Uniform(low, high)
