import tensorflow as tf
import numpy as np

def sample(probs):
    random_uniform = tf.random_uniform(tf.shape(probs))
    scaled_random_uniform = tf.log(random_uniform) / probs
    return tf.argmax(scaled_random_uniform, axis=1)

class Model:

    def __init__(self, policy, observation_space, action_space, learning_rate, spatial_resolution):
        self.spatial_resolution = spatial_resolution
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = policy
        self.learning_rate = learning_rate

        self.reward = tf.placeholder(tf.float32, [None], name="reward")
        self.action_spatial_mask = tf.placeholder(tf.float32, [None], name="action_spatial_mask")
        self.available_actions_mask = tf.placeholder(tf.float32, [None, self.action_space], name="available_actions")
        self.action_spatial_index = tf.placeholder(tf.int32, [None], name="action_spatial_index")
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.model = self.policy(self.observation_space, self.action_space, self.spatial_resolution)

        action_probs = self.model.policy * self.available_actions_mask

        # normalize to 1
        action_probs /= tf.reduce_sum(action_probs, axis=1, keepdims=True)

        action_log_probs = tf.reduce_sum(
            tf.one_hot(self.action, self.action_space) *
            tf.log(action_probs + 1e-13)
            , axis=1)

        action_spatial_log_probs = tf.reduce_sum(
            tf.one_hot(self.action_spatial_index, self.spatial_resolution[0] * self.spatial_resolution[1]) *
            tf.expand_dims(self.action_spatial_mask, axis=1) *
            tf.log(self.model.policy_spatial + 1e-13)
            , axis=1)

        self.sampled_action = sample(action_probs)
        self.sampled_action_spatial = sample(self.model.policy_spatial)

        self.loss = -tf.reduce_mean((action_spatial_log_probs + action_log_probs) * self.reward)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99).minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def predict_action(self, observations, available_actions):

        feed_dict = {
            self.available_actions_mask: available_actions,
            self.model.screen_player: np.asarray(observations[0]),
            self.model.screen_beacon: np.asarray(observations[1]),
            self.model.screen_selected: np.asarray(observations[2])
        }

        action_ids, spatial_action_indexes = self.session.run(
            [self.sampled_action, self.sampled_action_spatial],
            feed_dict=feed_dict
        )
        return action_ids, spatial_action_indexes

    def train(self, observations, actions, available_actions_mask, actions_spatial, actions_spatial_mask, rewards):
        feed_dict = {
            self.reward: rewards,
            self.action_spatial_mask: actions_spatial_mask,
            self.available_actions_mask: available_actions_mask,
            self.action_spatial_index: actions_spatial,
            self.action: actions,
            self.model.screen_player: np.asarray(observations[0]),
            self.model.screen_beacon: np.asarray(observations[1]),
            self.model.screen_selected: np.asarray(observations[2])
        }

        loss, _ = self.session.run([self.loss, self.optimizer], feed_dict=feed_dict)

    def save(self, id):
        variables = tf.trainable_variables()
        saver = tf.train.Saver(variables)
        saver.save(self.session, "saver/model_{}.ckpt".format(id), write_meta_graph=False)

    def load(self, id):
        variables = tf.trainable_variables()
        saver = tf.train.Saver(variables)
        saver.restore(self.session, "saver/model_{}.ckpt".format(id))