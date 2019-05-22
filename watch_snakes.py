import time
import pickle
import keyboard
from Board import Board

if __name__ == '__main__':
  # NOTE: make sure that Board is initialized the same in `teach_snakes.py`
  board = Board(30, 10, are_snakes_helpless=False, are_snakes_learning=True)

  with open('models/model.p', 'rb') as model_file:
    board.oracle.Q = pickle.load(model_file)

  num_frames = 2000
  delay = 0.05

  drawings = []  # TODO: Add back and forward buttons

  while True:  # The drawing loop.
    try:
      if keyboard.is_pressed('k'):  # Pause
        while True:
          try:
            if keyboard.is_pressed('i'):  # Un-pause
              break
            else:
              pass
          except KeyboardInterrupt:
            break
      else:
        drawing = board.get_drawing()
        drawings.append(drawing)
        board.print_drawing(drawing)
        board.tick()
        time.sleep(delay)
    except KeyboardInterrupt:
      break
    else:
      with open('models/model.p', 'wb') as model_file:
        pickle.dump(board.oracle.Q, model_file)

  # Helpful debugging stuff...

  try:
    # TODO: Check that snakes dying in the beginning are shorter than snakes dying in the end.
    print('Average length of dead snakes: %.2f' %
        (sum(board.lengths_of_dead_snakes)/len(board.lengths_of_dead_snakes)))
  except ZeroDivisionError:
    pass

  # Iterate over the keys of the Oracle (the states that the Oracle has seen)
  # and pick out the states associated with eating and not eating fruit.
  # State contains the action in the last slot (-1 in Python).
  eat_fruit_Q = [board.oracle.Q[v] for v in board.oracle.Q if v[1][-1] == 1]
  no_fruit_Q = [board.oracle.Q[v] for v in board.oracle.Q if v[1][-1] == 0]
  #TODO: it would be interesting to see the values of the states adjacent to the ones
  # associated with eating fruit.

  print('Average Q of eating states is %.2f' %
      (sum(eat_fruit_Q)/len(eat_fruit_Q)))

  print('Average Q of not eating states is %.2f' %
      (sum(no_fruit_Q) / len(no_fruit_Q)))

  print('Average Q of all states is %.2f' % (sum(board.oracle.Q.values())/len(board.oracle.Q)))

  print('Number of Q states is %i' % len(board.oracle.Q))
