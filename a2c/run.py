from a2c.runner import Runner
from a2c.model import Model
from a2c.policies import PolicyFullyConnected
from common.env_wrapper import EnvWrapper
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.features import Dimensions
from pysc2.lib.features import AgentInterfaceFormat
from common.utilities import global_seed
import argparse
import sys
import absl.flags
absl.flags.FLAGS(sys.argv)


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--timesteps', default=int(1e6))
    parser.add_argument('--num_steps', default=5)
    parser.add_argument('--discount_rate', default=0.99)

    parser.add_argument('--learning_rate', default=2e-4)
    parser.add_argument('--summary_frequency', default=20000)
    parser.add_argument('--performance_num_episodes', default=10)
    parser.add_argument('--summary_log_dir', default="a2c")
    args = parser.parse_args()

    dimensions = Dimensions(screen=(32, 32), minimap=(1, 1))
    interfaceFormat = AgentInterfaceFormat(
        feature_dimensions=dimensions,
        use_feature_units=True,
    )

    global_seed(0)

    env = SC2Env(map_name="MoveToBeacon",
                 agent_interface_format=interfaceFormat,
                 step_mul=8,
                 random_seed=1
                 )

    env = EnvWrapper(env)

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=args.learning_rate,
        spatial_resolution=(5, 5)
    )

    runner = Runner(
        env=env,
        model=model,
        batch_size=args.num_steps,
        discount_rate=args.discount_rate,
        summary_log_dir=args.summary_log_dir,
        summary_frequency=args.summary_frequency,
        performance_num_episodes=args.performance_num_episodes
    )

    for _ in range(0, (args.timesteps // args.num_steps) + 1):
        runner.run()


def main():
    run()


if __name__ == "__main__":
    main()
