import pickle
import matplotlib
matplotlib.use('TkAgg')  # Fixes "RuntimeError: Python is not installed as a framework."
import matplotlib.pyplot as plt
from Board import Board

if __name__ == '__main__':
  # NOTE: make sure that Board is initialized the same in `teach_snakes.py`
  board = Board(30, 20)

  num_frames = 10 ** 3

  board.oracle._print_Q_summary_snapshot()

  num_states = []
  reached_states_ratio = []
  avg_length_of_dead_snakes = []
  avg_Q_of_dead_snakes = []
  avg_length_of_dead_random_snakes = []
  avg_Q_of_dead_random_snakes = []
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
        avg_length_of_dead_random_snakes.append(qsummary.avg_length_of_dead_random_snakes)
        avg_Q_of_dead_random_snakes.append(qsummary.avg_Q_of_dead_random_snakes)
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

  color = 'red'
  ax1.set_xlabel('Frames (100\'s)')
  ax1.set_ylabel('Num of States Reached', color=color)
  ax1.plot(t, num_states, color=color, label='Num of States Reached')
  ax1.tick_params(axis='y', labelcolor=color)

  plt.legend(loc=0)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'green'
  ax2.set_ylabel('Q', color=color)  # we already handled the x-label with ax1
  ax2.plot(t, avg_Q_eating_states, color=color, label='Eating states')
  ax2.plot(t, avg_Q_one_away_eating_states, color='blue', label='One away from eating')
  ax2.plot(t, avg_Q_two_away_eating_states, color='cyan', label='Two away from eating')
  ax2.plot(t, avg_Q_no_eating_states, color='magenta', label='No eating states')
  ax2.plot(t, avg_Q, color='yellow', label='Average Q')
  ax2.tick_params(axis='y', labelcolor=color)

  plt.legend(loc=2)

  plt.savefig('data/Q_convergence_summary.png')

  fig, ax1 = plt.subplots()

  plt.xlabel('Frames (100\'s)')
  plt.ylabel('N')

  color = 'red'
  ax1.set_xlabel('Frames (100\'s)')
  ax1.set_ylabel('Average length of dead snakes', color=color)
  ax1.plot(t, avg_length_of_dead_snakes, color=color, label='Average length of dead snakes')
  ax1.plot(t, avg_length_of_dead_random_snakes, color='black', label='Average length of dead random snakes')
  ax1.tick_params(axis='y', labelcolor=color)

  plt.legend(loc=0)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'green'
  ax2.set_ylabel('Q', color=color)  # we already handled the x-label with ax1
  ax2.plot(t, avg_Q_of_dead_snakes, color=color, label='Average Q of dead snakes')
  ax2.plot(t, avg_Q_of_dead_random_snakes, color='black', label='Average Q of dead random snakes')
  ax2.tick_params(axis='y', labelcolor=color)

  plt.legend(loc=2)

  plt.savefig('data/snakes_convergence_summary.png')

  board.oracle._print_Q_summary_snapshot()

  with open('data/model.p', 'wb') as model_file:
    pickle.dump(board.oracle.Q, model_file)
