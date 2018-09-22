from ppo.runner import Runner
from ppo.model import Model
from ppo.policies import PolicyFullyConnected
from common.env_wrapper import EnvWrapper
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.features import Dimensions
from pysc2.lib.features import AgentInterfaceFormat
from common.utilities import global_seed
import argparse
import numpy as np
import sys
import absl.flags
absl.flags.FLAGS(sys.argv)


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--timesteps', default=int(1e6))
    parser.add_argument('--num_steps', default=128)
    parser.add_argument('--entropy_coefficient', default=0.01)
    parser.add_argument('--learning_rate', default=2e-4)
    parser.add_argument('--gae_gamma', default=0.99)
    parser.add_argument('--gae_lambda', default=0.95)
    parser.add_argument('--num_batches', default=4)
    parser.add_argument('--num_training_epochs', default=4)
    parser.add_argument('--clip_range', default=0.2)
    parser.add_argument('--summary_frequency', default=20000)
    parser.add_argument('--performance_num_episodes', default=10)
    parser.add_argument('--summary_log_dir', default="ppo_fc")
    args = parser.parse_args()

    dimensions = Dimensions(screen=(32, 32), minimap=(1, 1))
    interface_format = AgentInterfaceFormat(
        feature_dimensions=dimensions,
        use_feature_units=True,
    )

    global_seed(0)
    batch_size = args.num_steps // args.num_batches
    env = SC2Env(map_name="MoveToBeacon",
                 agent_interface_format=interface_format,
                 step_mul=8,
                 random_seed=1
                 )

    env = EnvWrapper(env)

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=args.learning_rate,
        spatial_resolution=(5, 5),
        clip_range=args.clip_range,
        entropy_coefficient=args.entropy_coefficient
    )

    runner = Runner(env=env,
                    model=model,
                    num_steps=args.num_steps,
                    advantage_estimator_gamma=args.gae_gamma,
                    advantage_estimator_lambda=args.gae_lambda,
                    summary_frequency=args.summary_frequency,
                    performance_num_episodes=args.performance_num_episodes,
                    summary_log_dir=args.summary_log_dir)

    for _ in range(0, (args.timesteps // args.num_steps) + 1):
        assert args.num_steps % args.num_batches == 0
        step = runner.run()
        observations = np.asarray(step[0])
        actions = np.asarray(step[1])
        available_actions = np.asarray(step[2])
        actions_spatial = np.asarray(step[3])
        actions_spatial_mask = np.asarray(step[4])
        advantage_estimations = np.asarray(step[5])
        values = np.asarray(step[6])
        probs = np.asarray(step[7])
        probs_spatial = np.asarray(step[8])
        indexes = np.arange(args.num_steps)

        for _ in range(args.num_training_epochs):
            np.random.shuffle(indexes)

            for i in range(0, args.num_steps, batch_size):
                shuffled_indexes = indexes[i:i + batch_size]
                model.train(observations=
                            [observations[0][shuffled_indexes],
                             observations[1][shuffled_indexes],
                             observations[2][shuffled_indexes]
                             ],
                            actions=actions[shuffled_indexes],
                            available_actions_mask=available_actions[shuffled_indexes],
                            actions_spatial=actions_spatial[shuffled_indexes],
                            actions_spatial_mask=actions_spatial_mask[shuffled_indexes],
                            advantages=advantage_estimations[shuffled_indexes],
                            values=values[shuffled_indexes],
                            probs=probs[shuffled_indexes],
                            probs_spatial=probs_spatial[shuffled_indexes]
                            )


def main():
    run()


if __name__ == "__main__":
    main()
