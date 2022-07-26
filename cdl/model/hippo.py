# Modify from https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/eRL_demo_PPOinSingleFile.py
"""net.py"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical

from model.hippo_skills import SkillController
from utils.utils import to_numpy, preprocess_obs, postprocess_obs

EPS = 1e-6


class dummy_module(nn.Module):
    # always return the preset dummy value
    def __init__(self, dummy_value):
        super().__init__()
        self.dummy_value = dummy_value

    def forward(self, *args):
        return self.dummy_value


class ActorPPO(nn.Module):
    def __init__(self, encoder, skills, params):
        super().__init__()
        self.encoder = encoder
        self.device = device = params.device
        ppo_params = params.policy_params.ppo_params
        self.hippo_params = hippo_params = params.policy_params.hippo_params

        self.log_std_min = ppo_params.log_std_min
        self.log_std_max = ppo_params.log_std_max

        feature_dim = encoder.feature_dim

        # init skill selection network
        skill_net = []
        self.num_skills = num_skills = len(skills)
        input_dim = feature_dim
        for output_dim in hippo_params.skill_net_dims:
            skill_net.append(nn.Linear(input_dim, output_dim))
            skill_net.append(nn.ReLU())
            input_dim = output_dim
        skill_net.append(nn.Linear(input_dim, num_skills))
        self.skill_net = nn.Sequential(*skill_net)

        # init object selection network
        self.num_objs = num_objs = hippo_params.num_objs
        obj_placeholder = torch.zeros(num_objs + 1, dtype=torch.float32, device=device)
        self.obj_nets = nn.ModuleList()
        for skill in skills:
            if skill.is_object_oriented_skill:
                obj_net_i = []
                input_dim = feature_dim
                for output_dim in hippo_params.obj_net_dims:
                    obj_net_i.append(nn.Linear(input_dim, output_dim))
                    obj_net_i.append(nn.ReLU())
                    input_dim = output_dim
                obj_net_i.append(nn.Linear(input_dim, num_objs + 1))
                obj_net_i = nn.Sequential(*obj_net_i)
                self.obj_nets.append(obj_net_i)
            else:
                self.obj_nets.append(dummy_module(obj_placeholder))
        self.mask_last_obj = [skill.is_object_oriented_skill and skill.is_object_necessary for skill in skills]

        # init skill param network
        self.num_max_skill_params = hippo_params.num_max_skill_params
        skill_param_placeholder = torch.zeros(self.num_max_skill_params, dtype=torch.float32, device=device)
        self.skill_param_nets = nn.ModuleList()
        for skill in skills:
            num_skill_params = skill.num_skill_params
            if num_skill_params:
                skill_param_net_i = []
                input_dim = feature_dim + num_objs + 1
                for output_dim in hippo_params.skill_params_net_dims:
                    skill_param_net_i.append(nn.Linear(input_dim, output_dim))
                    skill_param_net_i.append(nn.ReLU())
                    input_dim = output_dim
                skill_param_net_i.append(nn.Linear(input_dim, num_skill_params * 2))
                skill_param_net_i = nn.Sequential(*skill_param_net_i)
                self.skill_param_nets.append(skill_param_net_i)
            else:
                self.skill_param_nets.append(dummy_module(skill_param_placeholder))

    def forward(self, state, deterministic=False, detach_encoder=False):
        feature = self.encoder(state, detach=detach_encoder)
        assert feature.ndim == 1

        # select skill
        skill_logits = self.skill_net(feature)                                              # (num_skills,)
        skill_dist = Categorical(logits=skill_logits)
        skill = skill_logits.argmax(dim=-1) if deterministic else skill_dist.sample()       # scalar
        skill_log_prob = skill_dist.log_prob(skill)                                         # scalar

        # select object to interact
        obj_net = self.obj_nets[skill]
        if isinstance(obj_net, dummy_module):
            obj_one_hot = obj_net()                                                         # (num_objs + 1,)
            obj_log_prob = 0
        else:
            obj_logits = obj_net(feature)                                                   # (num_objs + 1,)
            if self.mask_last_obj[skill]:
                obj_logits[-1] = float('-inf')
            obj_dist = OneHotCategorical(logits=obj_logits)
            if deterministic:
                obj_one_hot = F.one_hot(obj_logits.argmax(-1), obj_logits.shape[-1]).float()
            else:
                obj_one_hot = obj_dist.sample()                                             # (num_objs + 1,)
            obj_log_prob = obj_dist.log_prob(obj_one_hot)                                   # scalar
        obj = obj_one_hot.argmax()                                                          # scalar

        # select skill parameters
        skill_param_net = self.skill_param_nets[skill]
        if isinstance(skill_param_net, dummy_module):
            skill_param = skill_param_net()                                                 # (num_max_skill_params,)
            skill_param_log_prob = 0
        else:
            skill_param = skill_param_net(torch.cat([feature, obj_one_hot]))                # (2 * num_skill_params,)
            skill_param_mean, skill_param_log_std = torch.split(skill_param, len(skill_param) // 2, dim=-1)
            skill_param_log_std = torch.clip(skill_param_log_std, self.log_std_min, self.log_std_max)
            skill_param_std = skill_param_log_std.exp()
            skill_param_dist = Normal(skill_param_mean, skill_param_std)
            if deterministic:
                skill_param = skill_param_mean
            else:
                skill_param = skill_param_dist.sample()                                     # (num_skill_params,)
            skill_param_log_prob = skill_param_dist.log_prob(skill_param).sum()             # scalar
        skill_param = torch.tanh(skill_param)

        log_prob = skill_log_prob + obj_log_prob + skill_param_log_prob                     # scalar

        # print("skill", skill_dist.probs)
        # print("skill", self.hippo_params.skill_names[skill])
        # if not isinstance(obj_net, dummy_module):
        #     print("obj_dist", obj_dist.probs)
        # print("obj", obj)
        # print("skill_param", skill_param)

        return skill, obj, skill_param, log_prob

    def get_log_prob_entropy(self, state, skill, obj, skill_param, detach_encoder=False):
        feature = self.encoder(state, detach=detach_encoder)                                    # (bs, feature_dim)
        assert feature.ndim == 2
        skill_logits = self.skill_net(feature)                                                  # (bs, num_skills)

        log_prob = []
        entropy = []
        for feature_i, skill_logits_i, skill_i, obj_i, skill_param_i in \
                zip(feature, skill_logits, skill, obj, skill_param):
            # skill
            skill_dist = Categorical(logits=skill_logits_i)
            skill_log_prob = skill_dist.log_prob(skill_i)                                       # scalar
            skill_sample = skill_dist.sample()                                                  # scalar
            skill_entropy = skill_dist.entropy()                                                # scalar

            # object
            obj_net = self.obj_nets[skill_i]
            if isinstance(obj_net, dummy_module):
                obj_log_prob = 0
            else:
                obj_logits = obj_net(feature_i)                                                 # (num_objs + 1,)
                if self.mask_last_obj[skill_i]:
                    obj_logits[-1] = float('-inf')
                obj_dist = Categorical(logits=obj_logits)
                obj_log_prob = obj_dist.log_prob(obj_i)                                         # scalar

            obj_net = self.obj_nets[skill_sample]
            if isinstance(obj_net, dummy_module):
                obj_sample = obj_net()
                obj_entropy = 0
            else:
                obj_logits = obj_net(feature_i)                                                 # (num_objs + 1,)
                if self.mask_last_obj[skill_sample]:
                    obj_logits[-1] = float('-inf')
                obj_dist = OneHotCategorical(logits=obj_logits)
                obj_sample = obj_dist.sample()
                obj_entropy = obj_dist.entropy()                                                # scalar

            # skill parameters
            skill_param_net = self.skill_param_nets[skill_i]
            if isinstance(skill_param_net, dummy_module):
                skill_param_log_prob = 0
            else:
                obj_one_hot = F.one_hot(obj_i.long(), self.num_objs + 1).float()
                skill_param = skill_param_net(torch.cat([feature_i, obj_one_hot]))
                skill_param_mean, skill_param_log_std = torch.split(skill_param, len(skill_param) // 2, dim=-1)
                skill_param_std = torch.clip(skill_param_log_std, self.log_std_min, self.log_std_max).exp()
                skill_param_dist = Normal(skill_param_mean, skill_param_std)
                skill_param_i = skill_param_i.clamp(-1 + EPS, 1 - EPS)
                skill_param_i = torch.atanh(skill_param_i)
                skill_param_log_prob = skill_param_dist.log_prob(skill_param_i).sum()           # scalar

            skill_param_net = self.skill_param_nets[skill_sample]
            if isinstance(skill_param_net, dummy_module):
                skill_param_entropy = 0
            else:
                skill_param = skill_param_net(torch.cat([feature_i, obj_sample]))
                skill_param_mean, skill_param_log_std = torch.split(skill_param, len(skill_param) // 2, dim=-1)
                skill_param_std = torch.clip(skill_param_log_std, self.log_std_min, self.log_std_max).exp()
                skill_param_dist = Normal(skill_param_mean, skill_param_std)
                sample = skill_param_dist.sample()                                              # (num_skill_params,)
                sample_log_prob = skill_param_dist.log_prob(sample).sum()                       # scalar
                sample = torch.tanh(sample)                                                     # (num_skill_params,)
                sample_log_prob -= torch.log(F.relu(1 - sample.pow(2)) + 1e-6).sum()            # scalar
                skill_param_entropy = -sample_log_prob

            log_prob_i = skill_log_prob + obj_log_prob + skill_param_log_prob
            entropy_i = skill_entropy + obj_entropy + skill_param_entropy
            log_prob.append(log_prob_i)
            entropy.append(entropy_i)
        return torch.stack(log_prob), torch.stack(entropy)


class CriticAdv(nn.Module):
    def __init__(self, encoder, params):
        super().__init__()
        self.encoder = encoder
        hippo_params = params.policy_params.hippo_params

        layers = []
        feature_dim = encoder.feature_dim
        input_dim = feature_dim
        for output_dim in hippo_params.critic_net_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state, detach_encoder=False):
        feature = self.encoder(state, detach=detach_encoder)
        return self.net(feature)


class HiPPO(nn.Module):
    def __init__(self, encoder, inference, params):
        super().__init__()
        self.encoder = encoder
        self.inference = inference

        self.params = params
        self.policy_params = policy_params = params.policy_params
        self.ppo_params = ppo_params = policy_params.ppo_params
        self.hippo_params = hippo_params = policy_params.hippo_params

        cenv_params = params.env_params.causal_env_params
        num_objs = cenv_params.num_movable_objects + cenv_params.num_unmovable_objects + cenv_params.num_random_objects
        self.num_objs = hippo_params.num_objs = num_objs

        self.sc = SkillController(params)
        self.skills = skills = self.sc.skills
        hippo_params.num_max_skill_params = max([skill.num_skill_params for skill in skills])
        self.uniform_action = getattr(hippo_params, "uniform_action", False)

        self.device = params.device
        self.num_env = params.env_params.num_env
        self.is_vecenv = self.num_env > 1

        self.discount = policy_params.discount

        self.ratio_clip = ppo_params.ratio_clip             # 0.2, ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = ppo_params.lambda_entropy     # 0.02, could be 0.01~0.05
        self.lambda_gae_adv = ppo_params.lambda_gae_adv     # 0.98; Generalized Advantage Estimation, 0.95~0.99
        self.if_use_gae = ppo_params.if_use_gae
        self.target_step = ppo_params.target_step
        self.batch_size = ppo_params.batch_size

        self.criterion = torch.nn.SmoothL1Loss()

        self.trajectory_list = []
        self.hl_trajectory_list = []
        self.ready_trajectory_list = []
        self.ready_hl_trajectory_list = []
        self.skill_done = True
        if self.is_vecenv:
            self.trajectory_list = [[] for _ in range(self.num_env)]
            self.hl_trajectory_list = [[] for _ in range(self.num_env)]
            self.skill_done = [True for _ in range(self.num_env)]
        self.get_reward_sum = self.get_reward_sum_gae if self.if_use_gae else self.get_reward_sum_raw

        self.actor = ActorPPO(encoder, skills, params).to(self.device)
        self.critic = CriticAdv(encoder, params).to(self.device)

        self.lr = policy_params.lr
        self.act_optim = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.cri_optim = torch.optim.Adam(self.critic.parameters(), self.lr)

        action_low, action_high = params.action_spec
        self._action_bias, self._action_scale = (action_low + action_high) / 2, (action_high - action_low) / 2
        self._action_ten_bias = torch.as_tensor(self._action_bias, dtype=torch.float32, device=self.device)
        self._action_ten_scale = torch.as_tensor(self._action_scale, dtype=torch.float32, device=self.device)

        self.buffer = []

        self.load(params.training_params.load_policy, self.device)

    def setup_annealing(self, step):
        pass

    def reset(self, i=0):
        if self.is_vecenv:
            self.skill_done[i] = True
        else:
            self.skill_done = True

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs = postprocess_obs(preprocess_obs(obs, self.params))

            if self.is_vecenv:
                for i, done in enumerate(self.skill_done):
                    if not done:
                        self.hl_trajectory_list[i].append(None)
                        continue
                    if self.uniform_action:
                        skill = torch.randint(0, len(self.skills), ())
                        skill_ = self.skills[skill]
                        object_necessary = skill_.is_object_oriented_skill and skill_.is_object_necessary
                        obj_high = self.num_objs if object_necessary else self.num_objs + 1
                        obj = torch.randint(0, obj_high, ())
                        skill_param = torch.rand(skill_.num_skill_params) * 2 - 1
                    else:
                        obs_i = {k: torch.from_numpy(v[i]).to(self.device) for k, v in obs.items()}
                        skill, obj, skill_param, log_prob = self.actor(obs_i, deterministic)             # all scalar
                        self.hl_trajectory_list[i].append([skill, obj, skill_param, log_prob])
                    obs_i = {k: v[i] for k, v in obs.items()}
                    self.sc.update_config(i, obs_i, skill, obj, skill_param)
            else:
                if self.skill_done:
                    if self.uniform_action:
                        skill = torch.randint(0, len(self.skills), ())
                        skill_ = self.skills[skill]
                        object_necessary = skill_.is_object_oriented_skill and skill_.is_object_necessary
                        obj_high = self.num_objs if object_necessary else self.num_objs + 1
                        obj = torch.randint(0, obj_high, ())
                        skill_param = torch.rand(skill_.num_skill_params) * 2 - 1
                    else:
                        obs_ten = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
                        skill, obj, skill_param, log_prob = self.actor(obs_ten, deterministic)
                        self.hl_trajectory_list.append([skill, obj, skill_param, log_prob])
                    self.sc.update_config(0, obs, skill, obj, skill_param)
                else:
                    self.hl_trajectory_list.append(None)
            action, self.skill_done = self.sc.get_action(obs)

            return action

    def act_randomly(self):
        return self._action_bias + \
               self._action_scale * np.random.uniform(-1, 1, self._action_scale.shape)

    def update_trajectory_list(self, obs, action, done, next_obs, info):
        if self.uniform_action:
            return
        obs = postprocess_obs(preprocess_obs(obs, self.params))
        next_obs = postprocess_obs(preprocess_obs(next_obs, self.params))
        if self.is_vecenv:
            for i in range(self.num_env):
                obs_i = {key: val[i] for key, val in obs.items()}
                next_obs_i = {key: val[i] for key, val in next_obs.items()}
                if done[i]:
                    next_obs_i = postprocess_obs(preprocess_obs(info[i]["obs"], self.params))
                self.trajectory_list[i].append([obs_i, action[i], done[i], next_obs_i])
                if done[i]:
                    assert len(self.trajectory_list[i]) == len(self.hl_trajectory_list[i])
                    self.ready_trajectory_list.extend(self.trajectory_list[i])
                    self.ready_hl_trajectory_list.extend(self.hl_trajectory_list[i])
                    self.trajectory_list[i] = []
                    self.hl_trajectory_list[i] = []
        else:
            self.trajectory_list.append([obs, action, done, next_obs])
            if done:
                assert len(self.trajectory_list) == len(self.hl_trajectory_list)
                self.ready_trajectory_list.extend(self.trajectory_list)
                self.ready_hl_trajectory_list.extend(self.hl_trajectory_list)
                self.trajectory_list = []
                self.hl_trajectory_list = []

    def update_target(self):
        pass

    def zip_buf_state(self, buf_state):
        # from list of dict to dict of list, then np.stack to each value
        # not sure whether we need preprocess_obs
        buf_state = {k: torch.from_numpy(np.stack([dic[k] for dic in buf_state])).to(self.device).float()
                     for k in buf_state[0]}
        return buf_state

    def update(self):
        # these input args are not used, just to keep the same api
        if len(self.ready_trajectory_list) < self.target_step:
            return {}

        _trajectory = list(map(list, zip(*self.ready_trajectory_list)))  # 2D-list transpose
        ten_state = self.zip_buf_state(_trajectory[0])
        ten_action = torch.as_tensor(np.array(_trajectory[1]), dtype=torch.float32, device=self.device)
        ten_mask = (1.0 - torch.as_tensor(np.array(_trajectory[2]), dtype=torch.float32, device=self.device))
        ten_mask *= self.discount
        ten_next_state = self.zip_buf_state(_trajectory[3])

        inference_batch_size = self.params.inference_params.batch_size
        ten_reward = [self.inference.reward({k: v[i:i + inference_batch_size] for k, v in ten_state.items()},
                                            ten_action[i:i + inference_batch_size, None],
                                            {k: v[i:i + inference_batch_size, None] for k, v in ten_next_state.items()})
                      for i in range(0, len(self.ready_trajectory_list), inference_batch_size)]
        ten_reward = torch.cat(ten_reward, dim=0).flatten()

        with torch.no_grad():
            buf_len = len(ten_reward)
            buf_state, buf_reward, buf_mask = ten_state, ten_reward, ten_mask

            buf_value = [self.critic({k: v[i:i + self.batch_size] for k, v in buf_state.items()})
                         for i in range(0, buf_len, self.batch_size)]
            buf_value = torch.cat(buf_value, dim=0)

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)

        '''PPO: Surrogate objective of Trust Region'''
        obj_critics = []
        for _ in range(int(buf_len / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = {k: v[indices] for k, v in buf_state.items()}
            r_sum = buf_r_sum[indices]

            value = self.critic(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optim_update(self.cri_optim, obj_critic)
            obj_critics.append(obj_critic)

        hl_idx = [i for i, ele in enumerate(self.ready_hl_trajectory_list) if ele is not None]
        buf_len = len(hl_idx)
        buf_state = {k: v[hl_idx] for k, v in buf_state.items()}
        buf_advantage = buf_advantage[hl_idx]

        hl_traj = [ele for i, ele in enumerate(self.ready_hl_trajectory_list) if ele is not None]
        hl_traj = list(map(list, zip(*hl_traj)))  # 2D-list transpose
        buf_skill, buf_obj, buf_skill_param, buf_log_prob = hl_traj
        buf_log_prob = torch.stack(buf_log_prob)

        obj_actors = []
        obj_entropies = []
        for _ in range(int(buf_len / self.batch_size) + 1):
            indices = np.random.randint(0, buf_len, size=self.batch_size)

            state = {k: v[indices] for k, v in buf_state.items()}
            skill = [buf_skill[i] for i in indices]
            obj = [buf_obj[i] for i in indices]
            skill_param = [buf_skill_param[i] for i in indices]
            log_prob = buf_log_prob[indices]
            advantage = buf_advantage[indices]

            new_log_prob, obj_entropy = self.actor.get_log_prob_entropy(state, skill, obj, skill_param)
            ratio = (new_log_prob - log_prob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate - obj_entropy.mean() * self.lambda_entropy
            if not torch.isfinite(obj_actor):
                continue

            self.optim_update(self.act_optim, obj_actor)
            obj_actors.append(obj_actor)
            obj_entropies.append(obj_entropy.mean())

        # clear ready_trajectory_list
        self.ready_trajectory_list = []
        self.ready_hl_trajectory_list = []

        loss = {
            "critic_loss": torch.stack(obj_critics).mean(),
        }
        if len(obj_actors):
            loss["actor_loss"] = torch.stack(obj_actors).mean()
            loss["entropy"] = torch.stack(obj_entropies).mean()

        return loss

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "actor_optimizer": self.act_optim.state_dict(),
                    "critic_optimizer": self.cri_optim.state_dict()
                    }, path)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("HiPPO loaded", path)
            checkpoint = torch.load(path, map_location=device)
            curren_state_dict = self.state_dict()
            load_state_dict = checkpoint["model"]
            load_state_dict = {k: v for k, v in load_state_dict.items() if "inference" not in k}
            load_state_dict.update({k: v for k, v in curren_state_dict.items() if "inference" in k})
            self.load_state_dict(load_state_dict)
            self.act_optim.load_state_dict(checkpoint["actor_optimizer"])
            self.cri_optim.load_state_dict(checkpoint["critic_optimizer"])
