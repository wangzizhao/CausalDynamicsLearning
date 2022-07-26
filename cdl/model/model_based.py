import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.utils import to_numpy, preprocess_obs, postprocess_obs


class ActionDistribution:
    def __init__(self, params):
        self.action_dim = action_dim = params.action_dim
        self.continuous_action = params.continuous_state

        model_based_params = params.policy_params.model_based_params
        self.n_top_candidate = model_based_params.n_top_candidate

        n_horizon_step = model_based_params.n_horizon_step
        std_scale = model_based_params.std_scale
        device = params.device

        if self.continuous_action:
            mu = torch.zeros(n_horizon_step, action_dim, dtype=torch.float32, device=device)
            std = torch.ones(n_horizon_step, action_dim, dtype=torch.float32, device=device) * std_scale
            self.init_dist = Normal(mu, std)

            action_low, action_high = params.action_spec
            self.action_low_device = torch.tensor(action_low, dtype=torch.float32, device=device)
            self.action_high_device = torch.tensor(action_high, dtype=torch.float32, device=device)
        else:
            probs = torch.ones(n_horizon_step, action_dim, dtype=torch.float32, device=device)
            # probs will be normalized by Categorical, so no need to normalize it here
            self.init_dist = Categorical(probs=probs)

        self.dist = self.init_dist

    def reset(self):
        self.dist = self.init_dist

    def sample(self, shape):
        """
        :param shape: int or tuple
        :return: (*shape, n_horizon_step, action_dim) if self.continuous_action else (*shape, n_horizon_step, 1)
        """
        if isinstance(shape, int):
            shape = (shape,)
        actions = self.dist.sample(shape)
        if self.continuous_action:
            actions = self.postprocess_action(actions)
        else:
            actions = actions.unsqueeze(dim=-1)
        return actions

    def update(self, actions, rewards):
        """
        :param actions: (n_candidate, n_horizon_step, action_dim) if self.continuous_action
            else (n_candidate, n_horizon_step, 1)
        :param rewards: (n_candidate, n_horizon_step, 1)
        :return:
        """
        sum_rewards = rewards.sum(dim=(1, 2))                           # (n_candidate,)

        top_candidate_idxes = torch.argsort(-sum_rewards)[:self.n_top_candidate]
        top_actions = actions[top_candidate_idxes]                      # (n_top_candidate, n_horizon_step, action_dim)

        if self.continuous_action:
            mu = top_actions.mean(dim=0)                                # (n_horizon_step, action_dim)
            std = torch.std(top_actions - mu, dim=0, unbiased=False)    # (n_horizon_step, action_dim)
            std = torch.clip(std, min=1e-6)
            self.dist = Normal(mu, std)
        else:
            top_actions = top_actions.squeeze(dim=-1)                   # (n_top_candidate, n_horizon_step)
            top_actions = F.one_hot(top_actions, self.action_dim)       # (n_top_candidate, n_horizon_step, action_dim)
            probs = top_actions.sum(dim=0)                              # (n_horizon_step, action_dim)
            # probs will be normalized by Categorical, so no need to normalize it here
            self.dist = Categorical(probs=probs)

    def get_action(self):
        if self.continuous_action:
            action = self.dist.mean[0]
            action = self.postprocess_action(action)
        else:
            action = self.dist.probs[0].argmax()
        return to_numpy(action)

    @staticmethod
    def clip(val, min_val, max_val):
        return torch.min(torch.max(val, min_val), max_val)

    def postprocess_action(self, action):
        return self.clip(action, self.action_low_device, self.action_high_device)


class ModelBased(nn.Module):
    def __init__(self, encoder, inference, params):
        super(ModelBased, self).__init__()

        self.encoder = encoder
        self.inference = inference

        self.params = params
        self.device = device = params.device
        self.model_based_params = model_based_params = params.policy_params.model_based_params

        self.use_current_state = getattr(model_based_params, "use_current_state", True)
        self.abstraction_feature_idxes = None
        self.use_ground_truth = getattr(model_based_params, "use_ground_truth", False)
        if self.use_ground_truth:
            model_based_params.use_abstraction_feature = False
        self.use_abstraction_feature = model_based_params.use_abstraction_feature
        if self.use_abstraction_feature:
            abstraction_graph = inference.get_state_abstraction()
            self.abstraction_feature_idxes = list(abstraction_graph.keys())
            print("model-based policy learn with abstraction graph")
            for key, value in abstraction_graph.items():
                print("{}: {}".format(key, value))

        self.init_model()
        self.action_dist = ActionDistribution(params)
        if self.continuous_state:
            self.action_low, self.action_high = params.action_spec
            self.action_mean = (self.action_low + self.action_high) / 2
            self.action_scale = (self.action_high - self.action_low) / 2

            # self.action_low_device = torch.tensor(self.action_low, dtype=torch.float32, device=device)
            # self.action_high_device = torch.tensor(self.action_high, dtype=torch.float32, device=device)
            # self.workspace_low = torch.tensor([-1.0, -1.0, 0.82], dtype=torch.float32, device=device)
            # self.workspace_high = torch.tensor([1.0, 1.0, 1.3], dtype=torch.float32, device=device)

        self.n_horizon_step = model_based_params.n_horizon_step
        self.n_iter = model_based_params.n_iter
        self.n_candidate = model_based_params.n_candidate

        self.to(device)
        self.optimizer = optim.Adam(self.fcs.parameters(), lr=params.policy_params.lr)

        self.load(params.training_params.load_model_based, device)
        self.train()

    def init_model(self):
        params = self.params
        model_based_params = self.model_based_params

        self.continuous_state = continuous_state = params.continuous_state

        feature_dim = self.encoder.feature_dim
        self.feature_inner_dim = feature_inner_dim = self.encoder.feature_inner_dim
        if self.use_abstraction_feature:
            if continuous_state:
                feature_dim = len(self.abstraction_feature_idxes)
            else:
                feature_dim = np.sum(feature_inner_dim[self.abstraction_feature_idxes])
        else:
            if not continuous_state:
                feature_dim = np.sum(feature_inner_dim)

        self.action_dim = action_dim = params.action_dim

        self.goal_keys = params.goal_keys
        obs_spec = params.obs_spec
        for key in self.goal_keys:
            assert obs_spec[key].ndim == 1, "Cannot concatenate because goal key {} is not 1D".format(key)
        goal_dim = np.sum([len(obs_spec[key]) for key in self.goal_keys])

        self.goal_inner_dim = None
        if not continuous_state:
            self.goal_inner_dim = []
            if self.goal_keys:
                self.goal_inner_dim = np.concatenate([params.obs_dims[key] for key in self.goal_keys])
            goal_dim = np.sum(self.goal_inner_dim)

        goal_dim = goal_dim.astype(np.int32)

        in_dim = feature_dim + action_dim + goal_dim
        modules = []
        for out_dim, activation in zip(model_based_params.fc_dims, model_based_params.activations):
            modules.append(nn.Linear(in_dim, out_dim))
            if activation == "relu":
                activation = nn.ReLU()
            elif activation == "leaky_relu":
                activation = nn.LeakyReLU()
            elif activation == "tanh":
                activation = nn.Tanh()
            else:
                raise ValueError("Unknown activation: {}".format(activation))
            modules.append(activation)
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))

        self.fcs = nn.Sequential(*modules)

    def setup_annealing(self, step):
        pass

    def update_target(self,):
        pass

    def act_randomly(self):
        if self.continuous_state:
            return self.action_mean + self.action_scale * np.random.uniform(-1, 1, self.action_scale.shape)
        else:
            return np.random.randint(self.action_dim)

    def extract_goal_feature(self, obs):
        if not self.goal_keys:
            return None

        goal = torch.cat([obs[k] for k in self.goal_keys], dim=-1)
        if self.continuous_state:
            return goal
        else:
            goal = torch.unbind(goal, dim=-1)
            goal = [F.one_hot(goal_i.long(), goal_i_dim).float() if goal_i_dim > 1 else goal_i.unsqueeze(dim=-1)
                    for goal_i, goal_i_dim in zip(goal, self.goal_inner_dim)]
            return torch.cat(goal, dim=-1)

    def pred_reward(self, obs, action, detach_encoder=True):
        output_numpy = False
        if isinstance(action, np.ndarray):
            output_numpy = True
            if action.dtype != np.float32:
                action = action.astype(np.float32)
            action = torch.from_numpy(action).to(self.device)
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}

        reward_need_squeeze = False
        if action.ndim == 1:
            reward_need_squeeze = True
            obs = {k: v[None] for k, v in obs.items()}                              # (bs, obs_spec)
            action = action[None]                                                   # (bs, action_dim)

        feature = self.encoder(obs, detach=detach_encoder)
        feature = self.get_abstraction_feature(feature)
        goal_feature = self.extract_goal_feature(obs)

        pred_reward = self.pred_reward_from_feature(feature, action, goal_feature)

        if reward_need_squeeze:
            pred_reward = torch.squeeze(pred_reward)                                # scalar

        if output_numpy:
            pred_reward = to_numpy(pred_reward)

        return pred_reward

    def pred_reward_from_feature(self, feature, action, goal_feature):
        if not self.continuous_state:
            feature = torch.cat(feature, dim=-1)
            action = F.one_hot(action.squeeze(dim=-1), self.action_dim).float()     # (bs, action_dim)

        reward_input = [feature, action] if goal_feature is None else [feature, action, goal_feature]
        reward_input = torch.cat(reward_input, dim=-1)

        pred_reward = self.fcs(reward_input)
        return pred_reward

    def ground_truth_reward(self, feature, action, goal_feature):
        if not self.continuous_state:
            feature = torch.cat(feature, dim=-1)

        env_name = self.params.env_params.env_name
        if env_name == "Chemical":
            current_color = []
            target_color = []
            idx = 0
            for i, feature_inner_dim_i in enumerate(self.feature_inner_dim):
                if i % 3 == 0:
                    current_color.append(feature[..., idx:idx + feature_inner_dim_i])
                idx += feature_inner_dim_i

            idx = 0
            for i, goal_inner_dim_i in enumerate(self.goal_inner_dim):
                target_color.append(goal_feature[..., idx:idx + goal_inner_dim_i])
                idx += goal_inner_dim_i

            num_matches = 0
            for current_color_i, target_color_i in zip(current_color, target_color):
                match = (current_color_i == target_color_i).all(dim=-1, keepdim=True)
                num_matches = num_matches + match
            pred_reward = num_matches
        elif env_name == "Physical":
            assert (self.feature_inner_dim == self.goal_inner_dim).all()

            diff = None
            idx = 0
            for feature_inner_dim_i in self.feature_inner_dim:
                current_pos = feature[..., idx:idx + feature_inner_dim_i].argmax(dim=-1, keepdim=True)
                target_pos = goal_feature[..., idx:idx + feature_inner_dim_i].argmax(dim=-1, keepdim=True)
                diff_i = -torch.abs(current_pos - target_pos)
                if diff is None:
                    diff = diff_i
                else:
                    diff += diff_i
                idx += feature_inner_dim_i
            pred_reward = diff
        elif env_name == "CausalReach":
            eef_pos = feature[..., 0:3]
            goal_pos = goal_feature
            dist = torch.abs(eef_pos - goal_pos).sum(dim=-1, keepdim=True)
            pred_reward = 1 - torch.tanh(10 * dist)
        elif env_name == "CausalPush":
            reach_mult = 0.5
            push_mult = 1.0

            eef_pos = feature[..., 0:3]
            mov_pos = feature[..., 5:8]
            goal_pos = goal_feature

            dist1 = torch.norm(eef_pos - mov_pos, dim=-1, keepdim=True)
            dist2 = torch.norm(mov_pos - goal_pos, dim=-1, keepdim=True)
            pred_reward = (1 - torch.tanh(5.0 * dist1)) * reach_mult + (1 - torch.tanh(5.0 * dist2)) * push_mult
        elif env_name == "CausalPick":
            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 0.5
            max_dist = 1.1
            xy_max_dist = 1.0
            z_max_dist = 0.2

            eef_pos = feature[..., 0:3]
            mov_pos = feature[..., 5:8]
            goal_pos = goal_feature
            gripper_action = action[..., -1:]

            # dist1 = torch.norm(eef_pos - mov_pos, dim=-1, keepdim=True)
            # r_reach = reach_mult * (1 - torch.tanh(5.0 * dist1)) * (gripper_action < 0)
            #
            # in_grasp = (dist1 < 0.01) * (gripper_action > 0)
            # r_grasp = in_grasp * grasp_mult
            #
            # dist2 = torch.norm(mov_pos - goal_pos, dim=-1, keepdim=True)
            # r_lift = (1 - torch.tanh(5.0 * dist2)) * lift_mult * in_grasp

            dist1 = torch.abs(eef_pos - mov_pos).sum(dim=-1, keepdim=True)
            xy_dist = torch.abs(eef_pos - mov_pos)[..., :2].sum(dim=-1, keepdim=True)
            z_dist = torch.abs(eef_pos - mov_pos)[..., 2:]
            xy_close = xy_dist < 0.05
            dist_score = (xy_max_dist - xy_dist + (z_max_dist - z_dist) * xy_close) / (xy_max_dist + z_max_dist)
            r_reach = reach_mult * dist_score * (gripper_action < 0)

            in_grasp = (dist1 < 0.02) * (gripper_action > 0)
            r_grasp = in_grasp * grasp_mult

            dist2 = torch.abs(goal_pos - mov_pos).sum(dim=-1, keepdim=True)
            r_lift = lift_mult * (max_dist - dist2) / max_dist * in_grasp

            pred_reward = r_reach + r_grasp + r_lift
        elif env_name == "CausalStack":
            reach_mult = 0.1
            grasp_mult = 0.35
            lift_mult = 1.0
            stack_mult = 2.0

            lift_height = 0.95
            xy_max_dist = 1.0
            z_max_dist = 0.2

            eef_pos = feature[..., 0:3]
            eef_xy = feature[..., 0:2]
            eef_z = feature[..., 2:3]

            mov_pos = feature[..., 5:8]
            mov_xy = feature[..., 5:7]
            mov_z = feature[..., 7:8]

            unmov_pos = feature[..., 8:11]
            unmov_xy = feature[..., 8:10]
            unmov_z = feature[..., 10:11]

            gripper_open = action[..., -1:] < 0
            gripper_close = action[..., -1:] > 0

            dist1 = torch.abs(eef_pos - mov_pos).sum(dim=-1, keepdim=True)
            xy_dist = torch.abs(eef_xy - mov_xy).sum(dim=-1, keepdim=True)
            z_dist = torch.abs(eef_z - mov_z)
            xy_close = xy_dist < 0.05
            dist_score = (xy_max_dist - xy_dist + (z_max_dist - z_dist) * xy_close) / (xy_max_dist + z_max_dist)
            r_reach = reach_mult * dist_score * gripper_open

            in_grasp = (dist1 < 0.02) * gripper_close
            r_grasp = in_grasp * grasp_mult

            dist2 = torch.abs(mov_xy - unmov_xy).sum(dim=-1, keepdim=True)
            z_dist = torch.abs(lift_height - mov_z)
            z_high = mov_z > 0.85
            dist_score = (xy_max_dist - dist2) * z_high / xy_max_dist + lift_height - 0.8 - z_dist
            r_lift = lift_mult * dist_score * in_grasp

            z_dist = mov_z - unmov_z
            r_stack = stack_mult * (dist2 < 0.01) * (0.04 < z_dist) * gripper_open

            pred_reward = r_reach + r_grasp + r_lift + r_stack
        else:
            raise NotImplementedError

        return pred_reward

    def act(self, obs, deterministic=False):
        """
        :param obs: (obs_spec)
        """
        if not deterministic and not self.continuous_state:
            if np.random.rand() < self.model_based_params.action_noise_eps:
                return self.act_randomly()

        self.inference.eval()
        self.action_dist.reset()

        planner_type = self.model_based_params.planner_type
        if planner_type == "cem":
            action = self.cem(obs)
        else:
            raise ValueError("Unknown planner type: {}".format(planner_type))

        if not deterministic and self.continuous_state:
            action_noise = self.model_based_params.action_noise
            action_noise = np.random.normal(scale=action_noise, size=self.action_dim)
            action = np.clip(action + action_noise, self.action_low, self.action_high)

        if self.continuous_state:
            eef_pos = obs["robot0_eef_pos"]
            global_low, global_high = np.array([-0.35, -0.45, 0.82]), np.array([0.35, 0.45, 1.0])
            controller_scale = 0.05
            action[:3] = np.clip(action[:3],
                                 (global_low - eef_pos) / controller_scale,
                                 (global_high - eef_pos) / controller_scale)
            action = np.clip(action, self.action_low, self.action_high)

        return action

    def get_abstraction_feature(self, feature):
        if self.use_abstraction_feature:
            if self.continuous_state:
                feature = feature[:, self.abstraction_feature_idxes]
            else:
                feature = [feature[idx] for idx in self.abstraction_feature_idxes]
        return feature

    def repeat_feature(self, feature, shape):
        """
        :param feature: 1-dimensional state/goal feature or None (do nothing if it's None)
        :param shape: repeat shape
        :return:
        """
        if feature is None:
            return None
        if isinstance(shape, int):
            shape = [shape]
        if isinstance(feature, torch.Tensor):
            return feature.expand(*shape, -1)
        else:
            return [feature_i.expand(*shape, -1) for feature_i in feature]

    def concat_current_and_next_features(self, feature, next_feature):
        feature = self.get_abstraction_feature(feature)                 # (n_candidate, feature_dim)
        next_feature = next_feature                                     # (n_candidate, n_horizon_step - 1, feature_dim)
        if self.continuous_state:
            # (n_candidate, n_horizon_step, feature_dim)
            return torch.cat([feature[:, None], next_feature], dim=1)   # (n_candidate, n_horizon_step, feature_dim)
        else:
            return [torch.cat([feature_i[:, None], next_feature_i], dim=1)
                    for feature_i, next_feature_i in zip(feature, next_feature)]

    def cem(self, obs):
        # cross-entropy method
        n_candidate = self.n_candidate
        inference = self.inference

        with torch.no_grad():
            obs = postprocess_obs(preprocess_obs(obs, self.params))
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
            feature = self.encoder(obs)

            # assumed the goal is fixed in the episode
            goal_feature = self.extract_goal_feature(obs)

            # (n_candidate, feature_dim)
            feature = self.repeat_feature(feature, n_candidate)
            # (n_candidate, n_horizon_step, goal_dim)
            goal_feature = self.repeat_feature(goal_feature, (n_candidate, self.n_horizon_step))

            for i in range(self.n_iter):
                actions = self.action_dist.sample(n_candidate)          # (n_candidate, n_horizon_step, action_dim)

                # (n_candidate, n_horizon_step, 1)
                if self.use_ground_truth:
                    pred_next_dist = inference.forward_with_feature(feature, actions, abstraction_mode=True)
                    pred_features = pred_next_dist.sample()
                    pred_rewards = self.ground_truth_reward(pred_features, actions, goal_feature)
                else:
                    if self.use_current_state:
                        if actions.shape[1] > 1:
                            pred_next_dist = inference.forward_with_feature(feature, actions[:, :-1],
                                                                            abstraction_mode=True)
                            pred_next_feature = inference.sample_from_distribution(pred_next_dist)
                            pred_features = self.concat_current_and_next_features(feature, pred_next_feature)
                        else:
                            # (n_candidate, n_horizon_step, feature_dim)
                            pred_features = feature[:, None, inference.abstraction_idxes]
                    else:
                        pred_next_dist = inference.forward_with_feature(feature, actions, abstraction_mode=True)
                        pred_next_feature = inference.sample_from_distribution(pred_next_dist)
                        pred_features = pred_next_feature

                    pred_rewards = self.pred_reward_from_feature(pred_features, actions, goal_feature)
                self.action_dist.update(actions, pred_rewards)

        return self.action_dist.get_action()                            # (action_dim,)

    def update(self, obs, action, reward):
        pred_reward = self.pred_reward(obs, action)
        pred_error = torch.abs(pred_reward - reward).squeeze(dim=-1)
        loss = pred_error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_detail = {"reward_pred_loss": loss,
                       "priority": to_numpy(pred_error)}
        return loss_detail

    def save(self, path):
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }, path)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "fcs" in k}
        own_state.update(state_dict)
        self.load_state_dict(own_state)

    def load(self, path, device):
        if path is not None and os.path.exists(path):
            print("ModelBased loaded", path)
            checkpoint = torch.load(path, map_location=device)
            self.load_my_state_dict(checkpoint["model"])                # only load reward predictor
            self.optimizer.load_state_dict(checkpoint["optimizer"])
