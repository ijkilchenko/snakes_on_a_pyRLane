import pickle
import matplotlib

matplotlib.use('TkAgg')  # Fixes "RuntimeError: Python is not installed as a framework."
import matplotlib.pyplot as plt
from Board import Board

if __name__ == '__main__':
  # NOTE: make sure that Board is initialized the same in `teach_snakes.py`
  board = Board(30, 10, are_snakes_random=False, are_snakes_learning=True)

  num_frames = 10 ** 5

  board.oracle._print_Q_summary_snapshot()

  num_states = []
  reached_states_ratio = []
  avg_length_of_dead_snakes = []
  avg_Q_of_dead_snakes = []
  avg_Q_eating_states = []
  avg_Q_one_away_eating_states = []
  avg_Q_two_away_eating_states = []
  avg_Q_no_eating_states = []
  avg_Q = []

  try:
    for i in range(num_frames):
      if i % 100 == 0:
        qsummary = board.oracle._get_Q_summary_snapshot()
        num_states.append(qsummary.num_states)
        reached_states_ratio.append(qsummary.reached_states_ratio)
        avg_length_of_dead_snakes.append(qsummary.avg_length_of_dead_snakes)
        avg_Q_of_dead_snakes.append(qsummary.avg_Q_of_dead_snakes)
        avg_Q_eating_states.append(qsummary.avg_Q_eating_states)
        avg_Q_one_away_eating_states.append(qsummary.avg_Q_one_away_eating_states)
        avg_Q_two_away_eating_states.append(qsummary.avg_Q_two_away_eating_states)
        avg_Q_no_eating_states.append(qsummary.avg_Q_no_eating_states)
        avg_Q.append(qsummary.avg_Q)

      board.tick()
  except KeyboardInterrupt:
    pass

  fig, ax1 = plt.subplots()

  t = [i for i in range(len(num_states))]

  plt.xlabel('Frames (100\'s)')
  plt.ylabel('N')

  color = 'tab:red'
  ax1.set_xlabel('Frames (100\'s)')
  ax1.set_ylabel('Num of States Reached', color=color)
  ax1.plot(t, num_states, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:green'
  ax2.set_ylabel('Q', color=color)  # we already handled the x-label with ax1
  ax2.plot(t, avg_Q_eating_states, color=color, label='Eating states')
  ax2.plot(t, avg_Q_one_away_eating_states, color='tab:blue', label='One away from eating')
  ax2.plot(t, avg_Q_two_away_eating_states, color='tab:cyan', label='Two away from eating')
  ax2.tick_params(axis='y', labelcolor=color)

  plt.legend()

  plt.savefig('data/Q_convergence_summary.png')

  board.oracle._print_Q_summary_snapshot()

  with open('data/model.p', 'wb') as model_file:
    pickle.dump(board.oracle.Q, model_file)
