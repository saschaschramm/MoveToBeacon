import time
from summary.logger import SummaryWriter


class StatsRecorder:

    def __init__(self, summary_frequency,
                 performance_num_episodes,
                 summary_log_dir,
                 save=False):
        self.total_rewards = []
        self.summary_frequency = summary_frequency
        self.performance_num_episodes = performance_num_episodes
        self.total_reward = 0
        self.num_episodes = 0
        self.start_time = 0
        self.summary_writer = SummaryWriter(summary_log_dir)
        self.save = save
        self.t = 0

    def print_score(self):
        end_time = time.time()
        if self.t == 0:
            elapsed_time = 0
        else:
            elapsed_time = end_time - self.start_time

        score = sum(self.total_rewards[-self.performance_num_episodes:]) / self.performance_num_episodes

        if self.save:
            self.summary_writer.write(value=score, step=self.t)

        print("{0:} {1:} {2:.1f}".format(self.t, score, elapsed_time))
        self.start_time = end_time

    def after_step(self, reward, terminal):
        self.total_reward += reward

        if self.t % self.summary_frequency == 0:
            self.print_score()

        if terminal:
            self.num_episodes += 1
            self.total_rewards.append(self.total_reward)
            self.total_reward = 0

        self.t += 1