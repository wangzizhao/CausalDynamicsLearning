import os
import time
import json
import torch
import shutil
import random
import numpy as np

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from env.physical_env import Physical
from env.chemical_env import Chemical
from utils.multiprocessing_env import SubprocVecEnv


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TrainingParams(AttrDict):
    def __init__(self, training_params_fname="params.json", train=True):
        config = json.load(open(training_params_fname))
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

        repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        training_params = self.training_params
        if getattr(training_params, "load_inference", None) is not None:
            training_params.load_inference = \
                os.path.join(repo_path, "interesting_models", training_params.load_inference)
        if getattr(training_params, "load_policy", None) is not None:
            training_params.load_policy = os.path.join(repo_path, "interesting_models", training_params.load_policy)
        if getattr(training_params, "load_model_based", None) is not None:
            training_params.load_model_based = \
                os.path.join(repo_path, "interesting_models", training_params.load_model_based)
        if getattr(training_params, "load_replay_buffer", None) is not None:
            training_params.load_replay_buffer = os.path.join(repo_path, "replay_buffer",
                                                              training_params.load_replay_buffer)

        if train:
            if training_params_fname == "policy_params.json":
                sub_dirname = "task" if training_params.rl_algo == "model_based" else "dynamics"
            else:
                raise NotImplementedError

            info = self.info.replace(" ", "_")
            experiment_dirname = info + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
            self.rslts_dir = os.path.join(repo_path, "rslts", sub_dirname, experiment_dirname)
            os.makedirs(self.rslts_dir)
            shutil.copyfile(training_params_fname, os.path.join(self.rslts_dir, "params.json"))

            self.replay_buffer_dir = None
            if training_params_fname == "policy_params.json" and training_params.replay_buffer_params.saving_freq:
                self.replay_buffer_dir = os.path.join(repo_path, "replay_buffer", experiment_dirname)
                os.makedirs(self.replay_buffer_dir)

        super(TrainingParams, self).__init__(self.__dict__)

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = self._clean_dict(v)
            _dict[k] = v
        return AttrDict(_dict)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def to_device(dictionary, device):
    """
    place dict of tensors + dict to device recursively
    """
    new_dictionary = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_dictionary[key] = to_device(val, device)
        elif isinstance(val, torch.Tensor):
            new_dictionary[key] = val.to(device)
        else:
            raise ValueError("Unknown value type {} for key {}".format(type(val), key))
    return new_dictionary


def preprocess_obs(obs, params):
    """
    filter unused obs keys, convert to np.float32 / np.uint8, resize images if applicable
    """
    def to_type(ndarray, type):
        if ndarray.dtype != type:
            ndarray = ndarray.astype(type)
        return ndarray

    obs_spec = getattr(params, "obs_spec", obs)
    new_obs = {}
    for k in params.obs_keys + params.goal_keys:
        val = obs[k]
        val_spec = obs_spec[k]
        if val_spec.ndim == 1:
            val = to_type(val, np.float32)
        if val_spec.ndim == 3:
            num_channel = val.shape[2]
            if num_channel == 1:
                env_params = params.env_params
                assert "Causal" in env_params.env_name
                val = to_type(val, np.float32)
            elif num_channel == 3:
                val = to_type(val, np.uint8)
            else:
                raise NotImplementedError
            val = val.transpose((2, 0, 1))                  # c, h, w
        new_obs[k] = val
    return new_obs


def postprocess_obs(obs):
    # convert images to float32 and normalize to [0, 1]
    new_obs = {}
    for k, val in obs.items():
        if val.dtype == np.uint8:
            val = val.astype(np.float32) / 255
        new_obs[k] = val
    return new_obs


def update_obs_act_spec(env, params):
    """
    get act_dim and obs_spec from env and add to params
    """
    params.continuous_state = params.continuous_action = params.continuous_factor = \
        not isinstance(env, (Physical, Chemical))
    if params.encoder_params.encoder_type == "conv":
        params.continuous_state = True

    params.action_dim = env.action_dim
    params.obs_spec = obs_spec = preprocess_obs(env.observation_spec(), params)

    if params.continuous_factor:
        params.obs_dims = None
        params.action_spec = env.action_spec
        # workspace_spec = env.workspace_spec()
        # workspace_spec = {k: v for k, v in workspace_spec.items() if k in params.obs_keys}
        # params.workspace_spec = workspace_spec
        # params.workspace_scale = workspace_scale = {k: v[1] - v[0] for k, v in workspace_spec.items()}
        # params.workspace_scale_array = np.concatenate([workspace_scale[k] for k in params.obs_keys], axis=-1)
    else:
        params.obs_dims = obs_dims = env.observation_dims()
        params.action_spec = None


def get_single_env(params, render=False):
    env_params = params.env_params
    env_name = env_params.env_name
    if "Causal" in env_name:
        causal_env_params = env_params.causal_env_params
        env = suite.make(env_name=env_params.env_name,
                         robots=causal_env_params.robots,
                         controller_configs=load_controller_config(default_controller=causal_env_params.controller_name),
                         gripper_types=causal_env_params.gripper_types,
                         has_renderer=render,
                         has_offscreen_renderer=causal_env_params.use_camera_obs,
                         use_camera_obs=causal_env_params.use_camera_obs,
                         camera_names=causal_env_params.camera_names,
                         camera_heights=causal_env_params.camera_heights,
                         camera_widths=causal_env_params.camera_widths,
                         camera_depths=causal_env_params.camera_depths,
                         ignore_done=False,
                         control_freq=causal_env_params.control_freq,
                         horizon=causal_env_params.horizon,
                         reward_scale=causal_env_params.reward_scale,
                         num_movable_objects=causal_env_params.num_movable_objects,
                         num_unmovable_objects=causal_env_params.num_unmovable_objects,
                         num_random_objects=causal_env_params.num_random_objects,
                         num_markers=causal_env_params.num_markers)
    elif env_name == "Physical":
        env = Physical(params)
    elif env_name == "Chemical":
        env = Chemical(params)
    else:
        raise ValueError("Unknown env_name: {}".format(env_name))

    return env


def get_subproc_env(params):
    def get_single_env_wrapper():
        return get_single_env(params)
    return get_single_env_wrapper


def get_env(params, render=False):
    num_env = params.env_params.num_env
    if render:
        assert num_env == 1
    if num_env == 1:
        return get_single_env(params, render)
    else:
        assert "Causal" in params.env_params.env_name, ""
        return SubprocVecEnv([get_subproc_env(params) for _ in range(num_env)])


def get_start_step_from_model_loading(params):
    """
    if model-based policy is loaded, return its training step;
    elif inference is loaded, return its training step;
    else, return 0
    """
    task_learning = params.training_params.rl_algo == "model_based"
    load_model_based = params.training_params.load_model_based
    load_inference = params.training_params.load_inference
    if load_model_based is not None and os.path.exists(load_model_based):
        model_name = load_model_based.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    elif load_inference is not None and os.path.exists(load_inference) and not task_learning:
        model_name = load_inference.split(os.sep)[-1]
        start_step = int(model_name.split("_")[-1])
        print("start_step:", start_step)
    else:
        start_step = 0
    return start_step


def get_object_interaction(obs, next_obs, is_demo, done, params):
    interactions_log = {}

    num_env = params.env_params.num_env
    if num_env == 1:
        if is_demo:
            return interactions_log
    else:
        not_valid = is_demo | done
        if not_valid.all():
            return interactions_log
        if not_valid.any():
            obs = {k: v[~not_valid] for k, v in obs.items()}
            next_obs = {k: v[~not_valid] for k, v in next_obs.items()}

    mov_pos_change = 0.02
    mov_euler_change = 2
    obj_touched = 0.1

    causal_env_params = params.env_params.causal_env_params
    eef_pos = obs["robot0_eef_pos"]
    eef_pos_next = next_obs["robot0_eef_pos"]
    for i in range(causal_env_params.num_movable_objects):
        obj_name = "mov{}".format(i)
        mov_pos = obs[obj_name + "_pos"]
        mov_pos_next = next_obs[obj_name + "_pos"]
        mov_euler = obs[obj_name + "_euler"]
        mov_euler_next = next_obs[obj_name + "_euler"]
        mov_moved = (np.linalg.norm(mov_pos_next - mov_pos, axis=-1) > mov_pos_change) | \
                    (np.linalg.norm(mov_euler_next - mov_euler, axis=-1) > mov_euler_change * np.pi / 180)
        mov_touched = (np.linalg.norm(eef_pos - mov_pos, axis=-1) < obj_touched) & \
                      (np.linalg.norm(eef_pos_next - mov_pos_next, axis=-1) < obj_touched)
        interactions_log[obj_name] = mov_moved & mov_touched
    for i in range(causal_env_params.num_unmovable_objects):
        obj_name = "unmov{}".format(i)
        unmov_pos = obs[obj_name + "_pos"]
        unmov_pos_next = next_obs[obj_name + "_pos"]
        unmov_touched = (np.linalg.norm(eef_pos - unmov_pos, axis=-1) < obj_touched) & \
                        (np.linalg.norm(eef_pos_next - unmov_pos_next, axis=-1) < obj_touched)
        interactions_log[obj_name] = unmov_touched
    for i in range(causal_env_params.num_random_objects):
        obj_name = "rand{}".format(i)
        rand_pos = obs[obj_name + "_pos"]
        rand_pos_next = next_obs[obj_name + "_pos"]
        rand_touched = (np.linalg.norm(eef_pos - rand_pos, axis=-1) < obj_touched) & \
                       (np.linalg.norm(eef_pos_next - rand_pos_next, axis=-1) < obj_touched)
        interactions_log[obj_name] = rand_touched

    return interactions_log
