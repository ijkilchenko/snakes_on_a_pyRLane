import pickle
from Board import Board

if __name__ == '__main__':
  # NOTE: make sure that Board is initialized the same in `teach_snakes.py`
  board = Board(30, 10, are_snakes_random=False, are_snakes_learning=True)

  num_frames = 10 ** 4

  board.oracle._print_Q_summary()

  try:
    for _ in range(num_frames):
      board.tick()
  except KeyboardInterrupt:
    pass

  with open('models/model.p', 'wb') as model_file:
    pickle.dump(board.oracle.Q, model_file)

  board.oracle._print_Q_summary()
