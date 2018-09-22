import tensorflow as tf


class PolicyFullyConnected:
    def __init__(self, observation_space, action_space, spatial_resolution):
        height, width = observation_space
        self.screen_player = tf.placeholder(tf.float32, [None, height, width], name="screen_player")
        self.screen_beacon = tf.placeholder(tf.float32, [None, height, width], name="screen_beacon")
        self.screen_selected = tf.placeholder(tf.float32, [None, height, width], name="screen_selected")

        inputs = tf.concat([
            self.screen_selected,
            self.screen_player,
            self.screen_beacon
        ], axis=2)

        channels = 3
        inputs_reshaped = tf.reshape(inputs, [tf.shape(inputs)[0], width * height * channels])

        hidden = tf.layers.dense(inputs=inputs_reshaped, units=256, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        logits_spatial = tf.layers.dense(inputs=hidden,
                                         units=spatial_resolution[0]*spatial_resolution[1],
                                         activation=None)

        self.probs = tf.nn.softmax(logits)
        self.probs_spatial = tf.nn.softmax(logits_spatial)