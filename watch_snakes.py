import time
import pickle
from Board import Board

if __name__ == '__main__':
  board = Board(30, 10, are_snakes_helpless=False)

  with open('models/model.p', 'rb') as model_file:
    board.oracle.Q = pickle.load(model_file)

  num_frames = 2000
  delay = 0.20

  try:
    for _ in range(num_frames):
      board.print()
      board.tick()
      time.sleep(delay)
  except KeyboardInterrupt:
    pass
  else:
    with open('models/model.p', 'wb') as model_file:
      pickle.dump(board.oracle.Q, model_file)

  try:
    print('Average length of dead snakes: %.2f' %
        (sum(board.lengths_of_dead_snakes)/len(board.lengths_of_dead_snakes)))
  except ZeroDivisionError:
    pass

  eat_fruit_Q = [board.oracle.Q[v] for v in board.oracle.Q if v[1][-1] == 1]
  no_fruit_Q = [board.oracle.Q[v] for v in board.oracle.Q if v[1][-1] == 0]

  print('Average Q of eating states is %.2f' %
      (sum(eat_fruit_Q)/len(eat_fruit_Q)))

  print('Average Q of not eating states is %.2f' %
      (sum(no_fruit_Q) / len(no_fruit_Q)))

  print('Average Q of all states is %.2f' % (sum(board.oracle.Q.values())/len(board.oracle.Q)))

  print('Number of Q states is %i' % len(board.oracle.Q))

