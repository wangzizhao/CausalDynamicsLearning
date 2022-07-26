"""
modified from
https://github.com/ARISE-Initiative/robosuite/blob/maple/robosuite/controllers/skill_controller.py
and
https://github.com/ARISE-Initiative/robosuite/blob/maple/robosuite/controllers/skills.py
"""

import numpy as np
from utils.utils import to_numpy


class BaseSkill:
    def __init__(self, skill_params):
        global_xyz_bounds = np.array(skill_params.global_xyz_bounds)
        self.global_param_bounds = global_xyz_bounds.copy()
        self.global_low = global_xyz_bounds[0]
        self.global_high = global_xyz_bounds[1]
        self.reach_threshold = skill_params.reach_threshold
        self.lift_height = skill_params.lift_height

        num_mov_objs = skill_params.num_movable_objects
        num_unm_objs = skill_params.num_unmovable_objects
        num_rand_objs = skill_params.num_random_objects
        obj_name_mapper = {}
        obj_name_mapper.update({i: "mov{}_pos".format(i) for i in range(num_mov_objs)})
        obj_name_mapper.update({i + num_mov_objs: "unmov{}_pos".format(i) for i in range(num_unm_objs)})
        obj_name_mapper.update({i + num_mov_objs + num_unm_objs: "rand{}_pos".format(i) for i in range(num_rand_objs)})
        self.obj_name_mapper = obj_name_mapper
        self.is_object_oriented_skill = False
        self.is_object_necessary = False
        self.num_objs = num_mov_objs + num_unm_objs + num_rand_objs

        self.controller_scale = skill_params.controller_scale

        self.init_obs = None
        self.obj = None
        self.params = None
        self.num_skill_params = 0
        self.num_max_step = 0
        self.num_step = 0

    def update_state(self, obs):
        pass

    def reset(self, obs, obj, params):
        self.init_obs = obs
        self.obj = obj
        self.params = params
        self.num_step = 0

    def get_obj_pos(self, obs):
        obj_name = self.obj_name_mapper.get(self.obj, None)
        if obj_name:
            return obs[obj_name].copy()
        else:
            return None

    def get_global_pos(self, scale):
        low, high = self.global_param_bounds
        return (high + low) / 2 + scale * (high - low) / 2

    def get_delta_pos(self, eef, goal_pos):
        return np.clip((goal_pos - eef) / self.controller_scale, -1, 1)

    def get_pos_ac(self, obs):
        raise NotImplementedError

    def get_gripper_ac(self, obs):
        raise NotImplementedError

    def is_success(self):
        return False

    def step(self, obs):
        self.update_state(obs)
        pos = self.get_pos_ac(obs)

        # eef after step shouldn't be out of global bound
        eef = obs["robot0_eef_pos"]
        delta_pos = self.controller_scale * pos
        delta_pos = np.clip(delta_pos, self.global_low - eef, self.global_high - eef)
        pos = np.clip(delta_pos / self.controller_scale, -1, 1)

        grp = self.get_gripper_ac(obs)
        self.num_step += 1
        return np.concatenate([pos, grp]), self.num_step >= self.num_max_step or self.is_success()


class AtomicSkill(BaseSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params)
        self.is_object_oriented_skill = False
        self.is_object_necessary = False
        self.num_skill_params = 4
        self.num_max_step = 1

    def get_pos_ac(self, obs):
        return self.params[:3].copy()

    def get_gripper_ac(self, obs):
        return self.params[3:].copy()


class GripperSkill(BaseSkill):
    def __init__(self, skill_params, is_open):
        super().__init__(skill_params)
        self.is_open = is_open
        self.is_object_oriented_skill = False
        self.is_object_necessary = False
        self.num_skill_params = 0
        self.num_max_step = skill_params.gripper_skill_params.num_max_step

    def get_pos_ac(self, obs):
        return np.zeros(3)

    def get_gripper_ac(self, info):
        gripper_action = np.ones(1)
        if self.is_open:
            gripper_action *= -1
        return gripper_action


class OpenSkill(GripperSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params, is_open=True)


class CloseSkill(GripperSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params, is_open=False)


class ReachSkill(BaseSkill):
    def __init__(self, skill_params):
        super().__init__(skill_params)
        reach_skill_params = skill_params.reach_skill_params
        if hasattr(reach_skill_params, "global_param_bounds"):
            self.global_param_bounds = np.array(reach_skill_params.global_param_bounds)

        self.is_object_oriented_skill = True
        self.is_object_necessary = False
        self.num_skill_params = 3

        self.num_max_step = reach_skill_params.num_max_step
        self.num_reach_steps = 0

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self.num_reach_steps = 0

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        goal_pos = self.get_obj_pos(self.init_obs)
        if goal_pos is None:
            goal_pos = self.get_global_pos(self.params)
        pos = self.get_delta_pos(eef, goal_pos)

        th = self.reach_threshold
        reached_xyz = (np.linalg.norm(eef - goal_pos) < th)
        self.num_reach_steps += reached_xyz

        return pos

    def get_gripper_ac(self, obs):
        return np.zeros(1)

    def is_success(self):
        return self.num_reach_steps >= 2


class GraspSkill(BaseSkill):

    STATES = ["INIT", "LIFTED", "HOVERING", "REACHED", "GRASPED"]

    def __init__(self, skill_params):
        super().__init__(skill_params)
        grasp_skill_params = skill_params.grasp_skill_params
        if hasattr(grasp_skill_params, "global_param_bounds"):
            self.global_param_bounds = np.array(grasp_skill_params.global_param_bounds)

        self.is_object_oriented_skill = True
        self.is_object_necessary = False
        self.num_skill_params = 3
        self.num_max_step = grasp_skill_params.num_max_step
        self.num_required_reach_steps = grasp_skill_params.num_reach_steps
            
        self.state = "INIT"
        self.num_grasp_steps = 0
        self.num_reach_steps = 0

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self.state = "INIT"
        self.num_grasp_steps = 0
        self.num_reach_steps = 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        goal_pos = self.get_obj_pos(obs)
        if goal_pos is None:
            goal_pos = self.get_global_pos(self.params)

        th = self.reach_threshold
        reached_lift = (eef[2] >= self.lift_height - th)
        reached_xy = (np.linalg.norm(eef[0:2] - goal_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(eef - goal_pos) < th)

        if self.state == "GRASPED" or \
                (self.state == "REACHED" and (self.num_reach_steps >= self.num_required_reach_steps)):
            self.state = "GRASPED"
            self.num_grasp_steps += 1
        elif self.state == "REACHED" or reached_xyz:
            self.state = "REACHED"
            self.num_reach_steps += 1
        elif reached_xy:
            self.state = "HOVERING"
        elif reached_lift:
            self.state = "LIFTED"
        else:
            self.state = "INIT"

        assert self.state in GraspSkill.STATES

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        goal_pos = self.get_obj_pos(obs)
        if goal_pos is None:
            goal_pos = self.get_global_pos(self.params)

        if self.state in ["INIT", "LIFTED"]:
            goal_pos[2] = self.lift_height

        pos = self.get_delta_pos(eef, goal_pos)

        return pos

    def get_gripper_ac(self, info):
        if self.state in ["GRASPED", "REACHED"]:
            gripper_action = np.ones(1)
        else:
            gripper_action = np.ones(1) * -1
        return gripper_action

    def is_success(self):
        return self.num_grasp_steps >= 2


class LiftSkill(BaseSkill):

    STATES = ["INIT", "LIFTED", "HOVERING", "REACHED", "GRASPED", "ARRIVED"]

    def __init__(self, skill_params):
        super().__init__(skill_params)
        lift_skill_params = skill_params.lift_skill_params
        if hasattr(lift_skill_params, "global_param_bounds"):
            self.global_param_bounds = np.array(lift_skill_params.global_param_bounds)

        self.is_object_oriented_skill = True
        self.is_object_necessary = True
        self.num_skill_params = 3
        self.num_max_step = lift_skill_params.num_max_step
        self.num_required_reach_steps = lift_skill_params.num_reach_steps
        self.num_required_grasp_steps = lift_skill_params.num_grasp_steps

        self.state = "INIT"
        self.num_grasp_steps = 0
        self.num_reach_steps = 0
        self.num_arrive_steps = 0

    def reset(self, *args, **kwargs):
        super().reset(*args, *kwargs)
        self.state = "INIT"
        self.num_grasp_steps = 0
        self.num_reach_steps = 0
        self.num_arrive_steps = 0
        if self.obj >= self.num_objs:
            self.obj = 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        obj_pos = self.get_obj_pos(obs)
        goal_pos = self.get_global_pos(self.params)
        assert obj_pos is not None

        th = self.reach_threshold
        reached_lift = (eef[2] >= self.lift_height - th)
        reached_xy = (np.linalg.norm(eef[0:2] - obj_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(eef - obj_pos) < th)
        reached_goal = (np.linalg.norm(eef - goal_pos) < th)

        if self.state == "ARRIVED" or \
                (self.state == "GRASPED" and self.num_grasp_steps >= self.num_required_grasp_steps):
            self.state = "ARRIVED"
            self.num_arrive_steps += reached_goal
        elif self.state == "GRASPED" or \
                (self.state == "REACHED" and (self.num_reach_steps >= self.num_required_reach_steps)):
            self.state = "GRASPED"
            self.num_grasp_steps += 1
        elif self.state == "REACHED" or reached_xyz:
            self.state = "REACHED"
            self.num_reach_steps += 1
        elif reached_xy:
            self.state = "HOVERING"
        elif reached_lift:
            self.state = "LIFTED"
        else:
            self.state = "INIT"

        assert self.state in LiftSkill.STATES

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        if self.state == "ARRIVED":
            goal_pos = self.get_global_pos(self.params)
        else:
            goal_pos = self.get_obj_pos(obs)
            assert goal_pos is not None

        if self.state in ["INIT", "LIFTED"]:
            goal_pos[2] = self.lift_height

        pos = self.get_delta_pos(eef, goal_pos)

        return pos

    def get_gripper_ac(self, info):
        if self.state in ["ARRIVED", "GRASPED", "REACHED"]:
            gripper_action = np.ones(1)
        else:
            gripper_action = np.ones(1) * -1
        return gripper_action

    def is_success(self):
        return self.num_arrive_steps >= 2


class PushSkill(BaseSkill):

    STATES = ["INIT", "LIFTED", "HOVERING", "REACHED", "PUSHED"]

    def __init__(self, skill_params):
        super().__init__(skill_params)
        push_skill_params = skill_params.push_skill_params
        if hasattr(push_skill_params, "global_param_bounds"):
            self.global_param_bounds = np.array(push_skill_params.global_param_bounds)

        self.is_object_oriented_skill = True
        self.is_object_necessary = True
        self.num_skill_params = 6
        self.num_max_step = push_skill_params.num_max_step
        self.delta_xyz_scale = np.array(push_skill_params.delta_xyz_scale)

        self.state = "INIT"
        self.num_reach_steps = 0

    def update_state(self, obs):
        eef = obs["robot0_eef_pos"]
        start, target = self.get_start_target()

        th = self.reach_threshold
        reached_lift = (eef[2] >= self.lift_height - th)
        reached_src_xy = (np.linalg.norm(eef[0:2] - start[0:2]) < th)
        reached_src_xyz = (np.linalg.norm(eef - start) < th)
        reached_target_xyz = (np.linalg.norm(eef - target) < th)

        if self.state in ["REACHED", "PUSHED"] and reached_target_xyz:
            self.state = "PUSHED"
            self.num_reach_steps += 1
        else:
            if self.state == "REACHED" or reached_src_xyz:
                self.state = "REACHED"
            else:
                if reached_src_xy:
                    self.state = "HOVERING"
                else:
                    if reached_lift:
                        self.state = "LIFTED"
                    else:
                        self.state = "INIT"

        assert self.state in PushSkill.STATES

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.state = "INIT"
        if self.obj >= self.num_objs:
            self.obj = 0
        self.num_reach_steps = 0

    def get_start_target(self):
        goal_pos = self.get_obj_pos(self.init_obs)
        assert goal_pos is not None
        start = np.clip(goal_pos + self.params[:3] * self.delta_xyz_scale, self.global_low, self.global_high)
        target = np.clip(goal_pos + self.params[3:] * self.delta_xyz_scale, self.global_low, self.global_high)
        return start, target

    def get_pos_ac(self, obs):
        eef = obs["robot0_eef_pos"]
        start, target = self.get_start_target()

        if self.state == "INIT":
            goal_pos = eef.copy()
            goal_pos[2] = self.lift_height
        elif self.state == "LIFTED":
            goal_pos = start.copy()
            goal_pos[2] = self.lift_height
        elif self.state == "HOVERING":
            goal_pos = start.copy()
        elif self.state == "REACHED":
            goal_pos = target.copy()
        elif self.state == "PUSHED":
            goal_pos = target.copy()
        else:
            raise NotImplementedError

        pos = self.get_delta_pos(eef, goal_pos)

        return pos

    def get_gripper_ac(self, obs):
        return np.ones(1)

    def is_success(self):
        return self.num_reach_steps >= 2


class SkillController:

    SKILL_DICT = {"atomic": AtomicSkill,
                  "reach": ReachSkill,
                  "grasp": GraspSkill,
                  "lift": LiftSkill,
                  "push": PushSkill,
                  "open": OpenSkill,
                  "close": CloseSkill}

    def __init__(self, params):
        self.params = params
        self.hippo_params = hippo_params = params.policy_params.hippo_params
        self.skill_params = skill_params = hippo_params.skill_params

        causal_params = params.env_params.causal_env_params
        skill_params.num_movable_objects = causal_params.num_movable_objects
        skill_params.num_unmovable_objects = causal_params.num_unmovable_objects
        skill_params.num_random_objects = causal_params.num_random_objects

        self.skills = [self.SKILL_DICT[skill_name](skill_params)
                       for skill_name in hippo_params.skill_names]

        self.num_env = params.env_params.num_env
        self.is_vecenv = self.num_env > 1
        if self.is_vecenv:
            self.skill_set = [[self.SKILL_DICT[skill_name](skill_params)
                               for skill_name in hippo_params.skill_names]
                              for _ in range(self.num_env)]
            self.cur_skill = [None for _ in range(self.num_env)]
        else:
            self.skill_set = self.skills
            self.cur_skill = None

    def update_config(self, i, obs, skill, obj, skill_param):
        skill = skill.item()
        obj = obj.item()
        skill_param = to_numpy(skill_param)

        if self.is_vecenv:
            self.cur_skill[i] = self.skill_set[i][skill]
            self.cur_skill[i].reset(obs, obj, skill_param)
        else:
            self.cur_skill = self.skill_set[skill]
            self.cur_skill.reset(obs, obj, skill_param)

    def get_action(self, obs):
        if self.is_vecenv:
            actions, dones = [], []
            for i in range(self.num_env):
                obs_i = {k: v[i] for k, v in obs.items()}
                action, done = self.cur_skill[i].step(obs_i)
                actions.append(action)
                dones.append(done)
            return np.array(actions), np.array(dones)
        else:
            return self.cur_skill.step(obs)
