import numpy as np
from summary.logger import SummaryWriter
from common.stats_recorder import StatsRecorder

from pysc2.lib import actions

class Runner():
    def __init__(self, env, model, batch_size, discount_rate,
                 summary_log_dir, save_summary_steps, performance_num_episodes):
        self.env = env
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.model = model

        self.file_writer = SummaryWriter(summary_log_dir)
        self.save_summary_steps = save_summary_steps
        self.performance_num_episodes = performance_num_episodes
        self.observations, self.available_actions_masks = self.env.reset()

        self.stats_recorder = StatsRecorder(summary_frequency=save_summary_steps,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)
        self.t = 0

    def discount(self, rewards, dones, discount_rate):
        discounted = []
        total_return = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                total_return = reward
            else:
                total_return = reward + discount_rate * total_return
            discounted.append(total_return)
        return np.asarray(discounted[::-1])

    def index_to_2d(self, action_spatial):
        position = np.unravel_index(action_spatial, self.model.spatial_resolution)
        if position[0] == 0:
            x = 0
        else:
            x = (position[0] * (self.env.observation_space[0] / (self.model.spatial_resolution[0] - 1))) - 1
        if position[1] == 0:
            y = 0
        else:
            y = (position[1] * (self.env.observation_space[0] / (self.model.spatial_resolution[1] - 1))) - 1
        return x, y

    def make_action_function(self, action, args):
        return actions.FunctionCall(action.id, args)

    def make_action(self, action_id, spatial_index):
        action = self.env.actions[action_id]
        if action == actions.FUNCTIONS.select_army:
            return actions.FUNCTIONS.select_army("select"), False
        elif action == actions.FUNCTIONS.Move_screen:
            x, y = self.index_to_2d(spatial_index)
            return self.make_action_function(actions.FUNCTIONS.Move_screen, [[0], [x, y]]), True
        else:
            raise NotImplementedError

    def run(self):
        batch_observations = []
        batch_rewards = []
        batch_actions = []
        batch_actions_spatial = []
        batch_actions_spatial_mask = []
        batch_available_actions = []
        batch_dones = []

        for _ in range(self.batch_size):
            batch_observations.append(self.observations)

            action_ids, spatial_indexes = self.model.predict_action(
                np.asarray([self.observations]).swapaxes(0, 1),
                [self.available_actions_masks]
            )

            action, spatial_mask = self.make_action(action_ids[0], spatial_indexes[0])
            batch_actions.append(action_ids[0])
            batch_actions_spatial.append(spatial_indexes[0])
            batch_actions_spatial_mask.append(spatial_mask)
            batch_available_actions.append(self.available_actions_masks)

            self.observations, rewards, dones, self.available_actions_masks = self.env.step(action)

            self.stats_recorder.after_step(reward=rewards, done=dones, t=self.t)
            self.t += 1

            batch_rewards.append(rewards)
            batch_dones.append(dones)

        batch_rewards = self.discount(batch_rewards, batch_dones, self.discount_rate)
        batch_observations = np.asarray(batch_observations).swapaxes(0, 1)

        self.model.train(batch_observations,
                         batch_actions,
                         batch_available_actions,
                         batch_actions_spatial,
                         batch_actions_spatial_mask,
                         batch_rewards)