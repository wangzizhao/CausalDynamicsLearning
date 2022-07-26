import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class IdentityEncoder(nn.Module):
    # extract 1D obs and concatenate them
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.keys = [key for key in params.obs_keys if params.obs_spec[key].ndim == 1]
        self.feature_dim = np.sum([len(params.obs_spec[key]) for key in self.keys])

        self.continuous_state = params.continuous_state
        self.feature_inner_dim = None
        if not self.continuous_state:
            self.feature_inner_dim = np.concatenate([params.obs_dims[key] for key in self.keys])

        self.to(params.device)

    def forward(self, obs, detach=False):
        if self.continuous_state:
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "manipulation_train", True):
                test_scale = self.manipulation_test_scale
                obs = {k: torch.randn_like(v) * test_scale if "marker" in k else v
                       for k, v in obs.items()}
            obs = torch.cat([obs[k] for k in self.keys], dim=-1)
            return obs
        else:
            obs = [obs_k_i
                   for k in self.keys
                   for obs_k_i in torch.unbind(obs[k], dim=-1)]
            obs = [F.one_hot(obs_i.long(), obs_i_dim).float() if obs_i_dim > 1 else obs_i.unsqueeze(dim=-1)
                   for obs_i, obs_i_dim in zip(obs, self.feature_inner_dim)]
            # overwrite some observations for out-of-distribution evaluation
            if not getattr(self, "chemical_train", True):
                assert self.params.env_params.env_name == "Chemical"
                assert self.params.env_params.chemical_env_params.continuous_pos
                test_scale = self.chemical_test_scale
                obs = [obs_i if obs_i.shape[-1] > 1 else torch.randn_like(obs_i) * test_scale for obs_i in obs]
            return obs


_AVAILABLE_ENCODERS = {"identity": IdentityEncoder}


def make_encoder(params):
    encoder_type = params.encoder_params.encoder_type
    return _AVAILABLE_ENCODERS[encoder_type](params)
