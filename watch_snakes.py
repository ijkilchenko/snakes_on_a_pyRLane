import time
import pickle
import keyboard
from Board import Board

class Controller:
  """Controls the board. You can rewind, pause/unpause, and fastforward the board
  using keys j, k, l, respectively. """

  def __init__(self, board):
    self.hook = keyboard.on_press_key('k', self.pause)
    keyboard.on_press_key('j', self.rewind)
    keyboard.on_press_key('l', self.fastforward)

    self.board = board

    self.last_drawing_index = -1
    self.drawings = []

    self.is_paused = False
    self.forward()

  def forward(self):
    while True:
      if not self.is_paused:
        drawing = self.board.get_drawing()
        self.drawings.append(drawing)

        self.board.print_drawing(drawing)

        self.last_drawing_index = len(self.drawings) - 1

        self.board.tick()
        time.sleep(delay)

  def resume(self, *args):
    keyboard.unhook(self.hook)
    self.hook = keyboard.on_press_key('k', self.pause)

    self.is_paused = False

  def pause(self, *args):
    keyboard.unhook(self.hook)
    self.hook = keyboard.on_press_key('k', self.resume)

    self.is_paused = True

  def rewind(self, *args):
    self.last_drawing_index += -1

    try:
      self.board.print_drawing(self.drawings[self.last_drawing_index])
    except IndexError:
      self.last_drawing_index = 0

    self.pause()

  def fastforward(self, *args):
    self.last_drawing_index += 1

    try:
      self.board.print_drawing(self.drawings[self.last_drawing_index])
    except IndexError:
      self.last_drawing_index = len(self.drawings)

    self.pause()

if __name__ == '__main__':
  # NOTE: make sure that Board is initialized the same in `teach_snakes.py`
  board = Board(30, 10, are_snakes_helpless=False, are_snakes_learning=True)

  try:
    with open('models/model.p', 'rb') as model_file:
      board.oracle.Q = pickle.load(model_file)
    print('Previous model loaded')
  except FileNotFoundError:
    print('Previous model not loaded!')

  num_frames = 12000  # 10 minutes if the delay is 0.05
  delay = 0.05

  try:
    controller = Controller(board)
  except KeyboardInterrupt:
    pass
  with open('models/model.p', 'wb') as model_file:
    pickle.dump(board.oracle.Q, model_file)

  board.oracle._print_Q_summary()

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
