import numpy as np
from summary.logger import SummaryWriter
from common.stats_recorder import StatsRecorder
from pysc2.lib import actions


class Runner:

    def __init__(self,
                 env,
                 model,
                 num_steps,
                 advantage_estimator_gamma,
                 advantage_estimator_lambda,
                 summary_frequency,
                 performance_num_episodes,
                 summary_log_dir
                 ):
        self.env = env
        self.model = model

        self.file_writer = SummaryWriter(summary_log_dir)
        self.performance_num_episodes = performance_num_episodes
        self.observation, self.available_actions_mask = self.env.reset()

        self.stats_recorder = StatsRecorder(summary_frequency=summary_frequency,
                                            performance_num_episodes=performance_num_episodes,
                                            summary_log_dir=summary_log_dir,
                                            save=True)

        self.gae_gamma = advantage_estimator_gamma
        self.gae_lambda = advantage_estimator_lambda
        self.terminal = False
        self.num_steps = num_steps
        self.advantage_estimation = 0


    def estimate_advantage(self, t, terminal, next_value):
        if terminal:
            delta = self.rewards[t] + self.values[t]
            return delta
        else:
            delta = self.rewards[t] + self.gae_gamma * next_value - self.values[t]
            return delta + self.gae_gamma * self.gae_lambda * self.advantage_estimation

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
        observations = []
        self.rewards = []
        actions = []
        actions_spatial = []
        actions_spatial_mask = []
        available_actions = []
        batch_dones = []
        self.values = []
        probs_spatial = []
        probs = []

        for _ in range(self.num_steps):
            observations.append(self.observation)

            action_ids, spatial_indexes, value, prob, prob_spatial = self.model.predict(
                np.asarray([self.observation]).swapaxes(0, 1),
                [self.available_actions_mask]
            )
            self.values.append(value)
            probs.append(prob)
            probs_spatial.append(prob_spatial)
            batch_dones.append(self.terminal)
            action, spatial_mask = self.make_action(action_ids[0], spatial_indexes[0])
            actions.append(action_ids[0])
            actions_spatial.append(spatial_indexes[0])
            actions_spatial_mask.append(spatial_mask)
            available_actions.append(self.available_actions_mask)

            self.observation, reward, self.terminal, self.available_actions_mask = self.env.step(action)
            self.stats_recorder.after_step(reward=reward, terminal=self.terminal)
            self.rewards.append(reward)

        advantage_estimations = np.zeros_like(self.rewards)
        last_value = self.model.predict_value(self.observation)[0]

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                self.advantage_estimation = self.estimate_advantage(t, self.terminal, last_value)
            else:
                self.advantage_estimation = self.estimate_advantage(t, batch_dones[t + 1], self.values[t + 1])
            advantage_estimations[t] = self.advantage_estimation

        observations = np.asarray(observations).swapaxes(0, 1)

        return observations, \
               actions, \
               available_actions, \
               actions_spatial, \
               actions_spatial_mask,\
               advantage_estimations,\
               self.values,\
               probs,\
               probs_spatial