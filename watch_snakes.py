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

  board.oracle._print_Q_summary()

  try:
    controller = Controller(board)
  except KeyboardInterrupt:
    pass
  with open('models/model.p', 'wb') as model_file:
    pickle.dump(board.oracle.Q, model_file)

  board.oracle._print_Q_summary()
