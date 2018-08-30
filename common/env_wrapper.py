import numpy as np
from pysc2.lib import actions
import numpy
numpy.set_printoptions(threshold=numpy.nan)

class EnvWrapper:

    def __init__(self, env):
        self.env = env
        self.height = 32
        self.width = 32

        self.actions = []
        self.actions.append(actions.FUNCTIONS.select_army)
        self.actions.append(actions.FUNCTIONS.Move_screen)

        self.action_space = len(self.actions)
        self.observation_space = (32, 32)

    def _available_actions_mask(self):
        available_actions = self.available_actions()
        available_actions_mask = np.zeros(self.action_space, dtype=int)
        for index, action in enumerate(self.actions):
            if action.id.value in available_actions:
                available_actions_mask[index] = 1
        return available_actions_mask

    def available_actions(self):
        return self.last_step.observation["available_actions"]

    def print_feature_units(self):
        for unit in self.last_step.observation.feature_units:
            print("unit", unit)
            print("alliance ", unit.alliance)
            print("x ", unit.x)
            print("y ", unit.y)

    def render(self):
        columns = []
        for i in range(32):
            rows = []
            for j in range(32):
                index = i * 32 + j
                if self.observation[index] == 1:
                    rows.append([255, 255, 255])
                else:
                    rows.append([0, 0, 0])
            columns.append(rows)
        self.viewer.imshow(np.asarray(columns, dtype=np.uint8))

    def reset(self):
        self.last_step = self.env.reset()[0]
        return self.extract_features(), self._available_actions_mask()

    def extract_features(self):
        player_relative = self.last_step.observation.feature_screen.player_relative
        player = (player_relative == 1).astype(int)
        beacon = (player_relative == 3).astype(int)

        observation = [
            player,
            beacon,
            self.last_step.observation.feature_screen.selected]
        return observation

    def step(self, action):
        self.last_step = self.env.step([action])[0]
        done = self.last_step.last()
        reward = self.last_step.reward
        return self.extract_features(), reward, done, self._available_actions_mask()