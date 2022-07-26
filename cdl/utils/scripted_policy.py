import random
import numpy as np

from env.physical_env import Coord


def get_is_demo(step, params):
    demo_annealing_start = params.scripted_policy_params.demo_annealing_start
    demo_annealing_end = params.scripted_policy_params.demo_annealing_end
    demo_annealing_coef = np.clip((step - demo_annealing_start) / (demo_annealing_end - demo_annealing_start), 0, 1)
    demo_prob_init = params.scripted_policy_params.demo_prob_init
    demo_prob_final = params.scripted_policy_params.demo_prob_final
    demo_prob = demo_prob_init + (demo_prob_final - demo_prob_init) * demo_annealing_coef
    return np.random.random() < demo_prob


class ScriptedPhysical:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params.action_dim

        self.num_objects = env.num_objects
        self.num_rand_objects = env.num_rand_objects
        self.width = env.width
        self.height = env.height
        self.directions = [Coord(-1, 0),
                           Coord(0, 1),
                           Coord(1, 0),
                           Coord(0, -1)]
        self.policy_id = 0
        self.reset()

    def reset(self, *args):
        policy_id = self.policy_id
        self.mov_obj_idx = policy_id // (self.num_objects + self.num_rand_objects - 1)
        self.target_obj_idx = policy_id % (self.num_objects + self.num_rand_objects - 1)
        if self.target_obj_idx >= self.mov_obj_idx:
            self.target_obj_idx += 1
        self.direction_idx = np.random.randint(4)
        self.direction = self.directions[self.direction_idx]
        self.success_steps = 0
        self.random_policy = np.random.rand() < 0.1

        n_policies = self.num_objects * (self.num_objects + self.num_rand_objects - 1)
        self.policy_id = (policy_id + 1) % n_policies

    def get_action(self, obj_idx, offset):
        if obj_idx >= self.num_objects:
            return 5 * np.random.randint(self.num_objects)
        return 5 * obj_idx + self.directions.index(offset) + 1

    def dijkstra(self, obj_idx_to_move, target_pos):
        env = self.env
        width, height = env.width, env.height
        Q = np.ones((width, height)) * np.inf
        dist = np.ones((width, height)) * np.inf
        checked = np.zeros((width, height), dtype=bool)
        for idx, obj in env.objects.items():
            checked[obj.pos.x, obj.pos.y] = True

        Q[target_pos.x, target_pos.y] = 0

        while True:
            x, y = np.unravel_index(np.argmin(Q), Q.shape)
            q = Q[x, y]
            if q == np.inf:
                break
            dist[x, y] = Q[x, y]
            checked[x, y] = True
            Q[x, y] = np.inf

            for del_x, del_y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                new_x, new_y = x + del_x, y + del_y
                if 0 <= new_x < width and 0 <= new_y < height and not checked[new_x, new_y]:
                    if q + 1 < Q[new_x, new_y]:
                        Q[new_x, new_y] = q + 1

        mov_obj = env.objects[obj_idx_to_move]
        mov_x, mov_y = mov_obj.pos.x, mov_obj.pos.y
        min_dist = np.inf
        min_idx = self.directions[0]
        for dir in self.directions:
            new_x, new_y = mov_x + dir.x, mov_y + dir.y
            if 0 <= new_x < width and 0 <= new_y < height:
                if dist[new_x, new_y] < min_dist:
                    min_dist = dist[new_x, new_y]
                    min_idx = dir
        return min_idx, min_dist

    def act(self, obs, deterministic=True):
        objects = self.env.objects
        env = self.env
        mov_obj_idx = self.mov_obj_idx
        target_obj_idx = self.target_obj_idx
        mov_obj = objects[mov_obj_idx]
        target_obj = objects[target_obj_idx]

        current_pos = mov_obj.pos
        target_pos = target_obj.pos - self.direction

        map_center = Coord(self.width // 2 + 1, self.height // 2 + 1)
        # need to push the target object from outside of the map (impossible), need to adjust the target object
        if not 0 <= target_pos.x < self.width or not 0 <= target_pos.y < self.height:
            if env.valid_move(target_obj_idx, self.direction):
                return self.get_action(target_obj_idx, self.direction)
            else:
                action_idx, dist = self.dijkstra(target_obj_idx, map_center)
                return self.get_action(target_obj_idx, action_idx)

        # unable to simply move the target object, need to plan a path for it instead (by letting it move to the center)
        pushed_pos = target_obj.pos + self.direction
        if any([obj.pos == pushed_pos for obj in objects.values() if obj != mov_obj]):
            action_idx, dist = self.dijkstra(target_obj_idx, map_center)
            if dist != np.inf:
                return self.get_action(target_obj_idx, action_idx)

        if current_pos != target_pos:
            if self.random_policy:
                return self.get_action(mov_obj_idx, random.choice(self.directions))
            else:
                action_idx, dist = self.dijkstra(mov_obj_idx, target_pos)
                if dist == np.inf:
                    return self.get_action(target_obj_idx, self.direction)
                else:
                    return self.get_action(mov_obj_idx, action_idx)
        else:
            action = self.get_action(mov_obj_idx, self.direction)
            self.success_steps += 1
            if self.success_steps > 3:
                self.reset()
            return action

    def act_randomly(self,):
        return np.random.randint(self.action_dim)


class ScriptedChemical:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.action_dim = params.action_dim

    def reset(self, *args):
        pass

    def act(self, obs, deterministic=True):
        return self.act_randomly()

    def act_randomly(self,):
        return np.random.randint(self.action_dim)
        # below for collecting dense graph only
        # p = np.concatenate([[i * 0.5 + 1.0] * 5 for i in range(10)])
        # p /= p.sum()
        # a = np.random.multinomial(1, p).argmax()
        # return a


class SingleScriptedPickAndPlace:
    def __init__(self, params):
        self.params = params

        causal_env_params = params.env_params.causal_env_params
        self.obj_to_pick_names = ["mov" + str(i) for i in range(causal_env_params.num_movable_objects)] + \
                                 ["unmov" + str(i) for i in range(causal_env_params.num_unmovable_objects)] + \
                                 ["rand" + str(i) for i in range(causal_env_params.num_random_objects)]

        pick_place_params = params.scripted_policy_params.pick_place_params
        self.release_prob = pick_place_params.release_prob
        self.noise_scale = pick_place_params.noise_scale
        self.action_scaling = pick_place_params.action_scaling
        self.push_prob = pick_place_params.push_prob
        self.push_z = pick_place_params.push_z
        self.random_ep_prob = pick_place_params.random_ep_prob
        self.rough_grasp_prob = pick_place_params.rough_grasp_prob
        self.rough_grasp_noise_scale = pick_place_params.rough_grasp_noise_scale
        self.rough_move_prob = pick_place_params.rough_move_prob

        self.is_stack = False
        self.is_demo = pick_place_params.is_demo
        if self.is_demo:
            self.is_stack = self.params.env_params.env_name == "CausalStack"
            self.obj_to_pick_names = self.obj_to_pick_names[:causal_env_params.num_movable_objects]
            self.release_prob = 0.0
            self.noise_scale = 0.0
            self.push_prob = 0.0
            self.random_ep_prob = 0.0
            self.rough_grasp_prob = 0.0
            self.rough_grasp_noise_scale = 0.0
            self.rough_move_prob = 0.0
            self.action_scaling = 20.0

        self.action_low, self.action_high = params.action_spec
        self.action_low *= 0.8
        self.action_high *= 0.8
        self.workspace_low, self.workspace_high = [-0.3, -0.4, 0.82], [0.3, 0.4, 1.00]

        self.episode_params_inited = False

    def reset(self, obs):
        self.obj_to_pick_name = random.choice(self.obj_to_pick_names)

        self.release = False
        self.release_step = 0
        self.push = np.random.rand() < self.push_prob
        self.random_ep = np.random.rand() < self.random_ep_prob

        self.rough_grasp = np.random.rand() < self.rough_grasp_prob
        self.rough_move = np.random.rand() < self.rough_move_prob
        self.goal = self.sample_goal(obs)

    def sample_goal(self, obs):
        if self.is_demo:
            if self.is_stack:
                goal = obs["unmov0_pos"].copy()
                goal[-1] = 0.95
                return goal
            else:
                return obs["goal_pos"]

        goal_key = "goal_" + self.obj_to_pick_name + "_pos"
        if goal_key in obs:
            return obs[goal_key]

        goal = np.random.uniform(self.workspace_low, self.workspace_high)
        if self.push:
            goal[2] = self.push_z

        # print("\nnew goal: {}\n".format(goal))

        return goal

    def act(self, obs, deterministic=True):
        if self.random_ep:
            return self.act_randomly()

        low, high = self.params.action_spec
        eef_pos = obs["robot0_eef_pos"]
        object_pos = obs[self.obj_to_pick_name + "_pos"]
        grasped = obs[self.obj_to_pick_name + "_grasped"]

        action = np.zeros_like(low)
        action[-1] = -1

        if not grasped:
            placement = object_pos - eef_pos
            xy_place, z_place = placement[:2], placement[-1]

            if np.abs(z_place) >= 0.1:
                action[:3] = placement
                action[-1] = np.random.rand() * 2 - 1
                noise_scale = self.rough_grasp_noise_scale
            else:
                if self.rough_grasp:
                    action[:3] = placement
                    action[-1] = np.random.rand() * 2.3 - 1.3
                    noise_scale = 0.2
                else:
                    if np.linalg.norm(xy_place) >= 0.02:
                        action[:2] = xy_place
                    elif np.abs(z_place) >= 0.02:
                        action[2] = z_place
                    elif np.linalg.norm(placement) >= 0.01:
                        action[:3] = placement
                    else:
                        action[-1] = 1
                        # print("try to grasp, success:", bool(grasped))
                    noise_scale = 0.0

        else:
            # eef position
            noise_scale = self.noise_scale

            placement = self.goal - eef_pos
            if np.linalg.norm(placement) < 0.1:
                self.goal = self.sample_goal(obs)

            if self.rough_move:
                action = self.act_randomly()
            else:
                action[:3] = placement

            # gripper
            to_release = np.random.rand() < self.release_prob
            if self.is_stack and np.linalg.norm(placement) < 0.02:
                to_release = True
            self.release = self.release or to_release
            if self.release:
                action[-1] = np.random.random() * 2 - 1
                self.release_step += 1
                if self.release_step >= 3:
                    self.release = False
                    self.release_step = 0
            else:
                action[-1] = np.random.random()

        action[:3] *= self.action_scaling
        noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=3)
        if self.push:
            noise[2] = np.clip(noise[2], -np.inf, 0.02)
        action[:3] += noise

        action[:3] = np.clip(action, low, high)[:3]

        return action

    def act_randomly(self):
        return np.random.uniform(self.action_low, self.action_high)


class ScriptedPickAndPlace:
    def __init__(self, params):
        self.num_env = params.env_params.num_env

        self.policies = [SingleScriptedPickAndPlace(params) for _ in range(self.num_env)]
        self.policies_inited = False

    def reset(self, obs, i=0):
        if not self.policies_inited:
            self.policies_inited = True
            for i in range(self.num_env):
                obs_i = {key: val[i] for key, val in obs.items()}
                self.policies[i].reset(obs_i)
        else:
            obs_i = {key: val[i] for key, val in obs.items()}
            self.policies[i].reset(obs_i)

    def act(self, obs):
        actions = []
        for i in range(self.num_env):
            obs_i = {key: val[i] for key, val in obs.items()}
            actions.append(self.policies[i].act(obs_i))
        return np.array(actions)

    def act_randomly(self):
        return self.policies[0].act_randomly()


def get_scripted_policy(env, params):
    env_name = params.env_params.env_name
    is_vecenv = params.env_params.num_env > 1
    if "Causal" in env_name:
        if is_vecenv:
            return ScriptedPickAndPlace(params)
        else:
            return SingleScriptedPickAndPlace(params)
    elif env_name == "Physical":
        return ScriptedPhysical(env, params)
    elif env_name == "Chemical":
        return ScriptedChemical(env, params)
    else:
        raise ValueError("Unknown env_name: {}".format(env_name))
