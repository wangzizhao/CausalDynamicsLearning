"""Gym environment for block pushing tasks (2D Shapes and 3D Cubes)."""

import numpy as np

import gym
from collections import OrderedDict
from dataclasses import dataclass
from gym.utils import seeding

import skimage
import matplotlib
import matplotlib.pyplot as plt

from env.drawing import diamond, square, triangle, cross, pentagon, parallelogram, scalene_triangle
from env.drawing import render_cubes, get_colors_and_weights


@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Coord(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)


@dataclass
class Object:
    pos: Coord
    weight: int


class InvalidMove(BaseException):
    pass


class InvalidPush(BaseException):
    pass


class Physical(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, params):
        self.params = params
        self.env_params = env_params = params.env_params
        self.physical_env_params = physical_env_params = env_params.physical_env_params

        self.width = physical_env_params.width
        self.height = physical_env_params.height
        self.render_type = physical_env_params.render_type
        self.mode = mode = physical_env_params.mode
        self.dense_reward = physical_env_params.dense_reward
        self.max_steps = physical_env_params.max_steps

        self.cmap = cmap = 'Blues'
        self.typ = 'Observed'
        self.new_colors = None

        if self.typ in ['Unobserved', 'FixedUnobserved'] and "FewShot" in mode:
            self.n_f = int(mode[-1])
            if cmap == 'Sets':
                self.new_colors = np.random.choice(12, self.n_f, replace=False)
            elif cmap == 'Pastels':
                self.new_colors = np.random.choice(8, self.n_f, replace=False)
            else:
                print("something went wrong")

        self.num_objects = num_objects = physical_env_params.num_objects
        self.num_rand_objects = physical_env_params.num_rand_objects
        self.num_actions = 5 * self.num_objects  # Move StayNESW
        self.num_weights = physical_env_params.num_weights
        if self.num_weights is None:
            self.num_weights = num_objects

        self.np_random = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.action_dim = self.num_actions

        # for rendering
        self.render_inited = False

        self.seed(params.seed)
        self.reset()

    def print(self):
        top_bottom_row = "-" * (self.width + 2)
        middle_row = "|" + " " * self.width + "|"
        map = [top_bottom_row] + [middle_row for _ in range(self.height)] + [top_bottom_row]
        for idx, obj in self.objects.items():
            x, y, w = obj.pos.x, obj.pos.y, obj.weight
            row = map[self.height - y]
            map[self.height - y] = row[:x + 1] + str(w) + row[x + 2:]
        for row in map:
            print(row)
        print()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_grid(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[idx][:3]
        return im

    def render_circles(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[idx][:3]
        return im.transpose([2, 0, 1])

    def render_shapes(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if self.shapes[idx] == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            elif self.shapes[idx] == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 2:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 3:
                rr, cc = diamond(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 4:
                rr, cc = cross(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 5:
                rr, cc = pentagon(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 6:
                rr, cc = parallelogram(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            else:
                rr, cc = scalene_triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]

            im[rr, cc, :] = self.colors[idx][:3]

        return im.transpose([2, 0, 1])

    def render_cubes(self):
        im = render_cubes(self.objects, self.width)
        return im.transpose([2, 0, 1])

    def render(self):
        if not self.render_inited:
            matplotlib.use("TkAgg")
            plt.ion()
            plt.figure(figsize=(6, 6))
            plt.show()
            self.render_inited = True
        image = dict(
            grid=self.render_grid,
            circles=self.render_circles,
            shapes=self.render_shapes,
            cubes=self.render_cubes,
        )[self.render_type]()
        image = image.transpose([2, 1, 0])
        plt.imshow(image)
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    def get_state(self):
        state = {}
        for idx, obj in self.objects.items():
            if idx < self.num_objects:
                name = "obj{}".format(idx)
            else:
                name = "rand{}".format(idx - self.num_objects)
            state[name] = np.array([obj.pos.x, obj.pos.y])

        for idx, obj in self.target_objects.items():
            if idx < self.num_objects:
                name = "target_obj{}".format(idx)
                state[name] = np.array([obj.pos.x, obj.pos.y])

        return state

    def observation_spec(self):
        return self.get_state()

    def observation_dims(self):
        state = {}
        for idx, obj in self.objects.items():
            if idx < self.num_objects:
                name = "obj{}".format(idx)
            else:
                name = "rand{}".format(idx - self.num_objects)
            state[name] = np.array([self.width, self.height])

        for idx, obj in self.target_objects.items():
            if idx < self.num_objects:
                name = "target_obj{}".format(idx)
                state[name] = np.array([self.width, self.height])

        return state

    def get_sparse_reward(self, target_objects):
        return float(self.get_dense_reward(target_objects) == 0)

    def get_dense_reward(self, target_objects):
        distance = 0.0
        for i in range(self.num_objects):
            distance += np.abs(self.objects[i].pos.x - target_objects[i].pos.x) + \
                        np.abs(self.objects[i].pos.y - target_objects[i].pos.y)

        distance /= self.num_objects
        return self.width + self.height - 2 - distance

    def reset(self, num_steps=30):
        self.cur_step = 0

        if self.typ == 'FixedUnobserved':
            self.shapes = np.arange(self.num_objects)
        elif self.mode == 'ZeroShotShape':
            self.shapes = np.random.choice(6, self.num_objects)
        else:
            self.shapes = np.random.choice(3, self.num_objects + self.num_rand_objects)

        self.objects = OrderedDict()
        if self.typ == 'Observed':
            self.colors, weights = get_colors_and_weights(
                cmap=self.cmap,
                num_colors=self.num_objects + self.num_rand_objects,
                observed=True,
                mode=self.mode)
        else:
            self.colors, weights = get_colors_and_weights(
                cmap=self.cmap,
                num_colors=self.num_objects,
                observed=False,
                mode=self.mode,
                new_colors=self.new_colors)
        weights[self.num_objects:] = [self.num_objects + 1] * self.num_rand_objects

        # Randomize object position.
        while len(self.objects) < self.num_objects + self.num_rand_objects:
            idx = len(self.objects)
            # Re-sample to ensure objects don't fall on same spot.
            while not (idx in self.objects and
                       self.valid_pos(self.objects[idx].pos, idx)):
                self.objects[idx] = Object(
                    pos=Coord(
                        x=np.random.choice(np.arange(self.width)),
                        y=np.random.choice(np.arange(self.height)),
                    ),
                    weight=weights[idx])

        self.target_objects = self.get_target(num_steps=num_steps)
        return self.get_state()

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if not 0 <= pos.x < self.width:
            return False
        if not 0 <= pos.y < self.height:
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True

    def valid_move(self, obj_id, offset: Coord):
        """Check if move is valid."""
        old_obj = self.objects[obj_id]
        new_pos = old_obj.pos + offset
        return self.valid_pos(new_pos, obj_id)

    def occupied(self, pos: Coord):
        for idx, obj in self.objects.items():
            if obj.pos == pos:
                return idx
        return None

    def translate(self, obj_id, offset: Coord, n_parents=0):
        """"Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) coordinate.
        """
        obj = self.objects[obj_id]

        other_object_id = self.occupied(obj.pos + offset)
        if other_object_id is not None:
            if n_parents == 1:
                # cannot push two objects
                raise InvalidPush()
            if obj.weight > self.objects[other_object_id].weight:
                self.translate(other_object_id, offset,
                               n_parents=n_parents+1)
            else:
                raise InvalidMove()
        if not self.valid_move(obj_id, offset):
            raise InvalidMove()

        self.objects[obj_id] = Object(
            pos=obj.pos+offset, weight=obj.weight)

    def step(self, action: int):
        directions = [Coord(0, 0),
                      Coord(-1, 0),
                      Coord(0, 1),
                      Coord(1, 0),
                      Coord(0, -1)]

        direction = action % 5
        obj_id = action // 5

        self.cur_step += 1
        done = self.cur_step >= self.max_steps

        for i in range(self.num_rand_objects):
            rand_obj_id = self.num_objects + i
            rand_direction = np.random.randint(4)
            try:
                self.translate(rand_obj_id, directions[rand_direction + 1])
            except (InvalidMove, InvalidPush):
                pass

        info = {'invalid_push': False}
        try:
            self.translate(obj_id, directions[direction])
        except InvalidMove:
            pass
        except InvalidPush:
            info['invalid_push'] = True

        if self.dense_reward:
            reward = self.get_dense_reward(self.target_objects)
            info["success"] = reward == 0
        else:
            reward = self.get_sparse_reward(self.target_objects)
            info["success"] = reward == 1

        state = self.get_state()

        return state, reward, done, info

    def sample_step(self, action: int):
        directions = [Coord(0, 0),
                      Coord(-1, 0),
                      Coord(0, 1),
                      Coord(1, 0),
                      Coord(0, -1)]

        direction = action % 5
        obj_id = action // 5
        done = False
        info = {'invalid_push': False}

        objects = self.objects.copy()
        try:
            self.translate(obj_id, directions[direction])
        except InvalidMove:
            pass
        except InvalidPush:
            info['invalid_push'] = True

        reward = self.get_dense_reward(self.target_objects)
        next_obs = self.render()
        self.objects = objects

        return reward, next_obs

    def get_target(self, num_steps):
        self.target_objects = objects = self.objects.copy()

        for i in range(num_steps):
            move = np.random.choice(self.num_objects * 5)
            self.step(move)

        target_objects = self.objects.copy()
        self.objects = objects

        return target_objects
