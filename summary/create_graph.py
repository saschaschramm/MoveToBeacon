from summary.logger import SummaryReader
summary_reader = SummaryReader("reinforce_fc")
summary_reader.plot(left_limit=2e4, right_limit=1e6, bottom_limit=0, top_limit=25, save=True)