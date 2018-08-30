import sys
import absl.flags
absl.flags.FLAGS(sys.argv)

from runner import Runner
from model import Model
from policies import PolicyFullyConnected

from common.env_wrapper import EnvWrapper
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.features import Dimensions
from pysc2.lib.features import AgentInterfaceFormat

import numpy as np
import tensorflow as tf
import random

def run():
    dir = "reinforce_fc"

    dimensions = Dimensions(screen=(32, 32), minimap=(1, 1))
    interfaceFormat = AgentInterfaceFormat(
        feature_dimensions=dimensions,
        use_feature_units=True,
    )

    np.random.seed(0)
    tf.set_random_seed(0)
    random.seed(0)

    env = SC2Env(map_name="MoveToBeacon",
                 agent_interface_format=interfaceFormat,
                 step_mul=8,
                 random_seed=1
                 )

    env = EnvWrapper(env)

    model = Model(
        policy=PolicyFullyConnected,
        observation_space = env.observation_space,
        action_space = env.action_space,
        learning_rate = 2e-4,
        spatial_resolution = (5, 5)
    )

    runner = Runner(
        env = env,
        model = model,
        batch_size = 100,
        discount_rate = 0.99,
        summary_log_dir=dir,
        save_summary_steps=20000,
        performance_num_episodes=10
    )

    while True:
        runner.run()

    env.close()

def main():
    run()

if __name__ == "__main__":
    main()