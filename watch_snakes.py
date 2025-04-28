import time
import pickle
from pynput import keyboard
from Board import Board


class Controller:
  """Controls the board. You can rewind, pause/unpause, and fastforward the board
  using keys j, k, and l, respectively. """

  def __init__(self, board, num_frames, delay):
    self.board = board
    self.num_frames = num_frames
    self.delay = delay

    self.last_drawing_index = -1
    self.drawings = []

    self.is_paused = False
    
    # Set up keyboard listener
    self.listener = keyboard.Listener(
        on_press=self.on_press,
        on_release=self.on_release)
    self.listener.start()
    
    self.forward()

  def on_press(self, key):
    try:
      if key.char == 'k':
        if self.is_paused:
          self.resume()
        else:
          self.pause()
      elif key.char == 'j':
        self.rewind()
      elif key.char == 'l':
        self.fastforward()
      elif key.char == 'q':
        print("\nQuitting...")
        self.listener.stop()
        raise KeyboardInterrupt
    except AttributeError:
      pass

  def on_release(self, key):
    pass

  def forward(self):
    while True and len(self.drawings) <= self.num_frames:
      if not self.is_paused:
        drawing = self.board.get_drawing()
        self.drawings.append(drawing)

        self.board.print_drawing(drawing)

        self.last_drawing_index = len(self.drawings) - 1

        self.board.tick()
        time.sleep(self.delay)

  def resume(self):
    if self.is_paused:
      print("\nResuming playback...")
      self.is_paused = False

  def pause(self):
    if not self.is_paused:
      print("\nPaused. Press 'k' to resume, 'j' to rewind, 'l' to fast forward, 'q' to quit")
      self.is_paused = True

  def rewind(self):
    if len(self.drawings) == 0:
      print("\nNo previous states to rewind to")
      return

    self.last_drawing_index = max(0, self.last_drawing_index - 1)
    self.board.print_drawing(self.drawings[self.last_drawing_index])
    self.pause()

  def fastforward(self):
    if self.last_drawing_index >= len(self.drawings) - 1:
      print("\nNo future states to fast forward to")
      return

    self.last_drawing_index = min(len(self.drawings) - 1, self.last_drawing_index + 1)
    self.board.print_drawing(self.drawings[self.last_drawing_index])
    self.pause()


if __name__ == '__main__':
  # NOTE: make sure that Board is initialized the same in `teach_snakes.py`
  board = Board(30, 20)

  try:
    with open('data/model.p', 'rb') as model_file:
      board.oracle.Q = pickle.load(model_file)
    print('Previous model loaded')
  except FileNotFoundError:
    print('Previous model not loaded!')

  num_frames = 12000  # 10 minutes if the delay is 0.05
  delay = 0.05

  board.oracle._print_Q_summary_snapshot()

  print("\nControls:")
  print("j - Rewind")
  print("k - Pause/Resume")
  print("l - Fast Forward")
  print("q - Quit")
  print("\nStarting playback...")

  try:
    controller = Controller(board, num_frames, delay)
  except KeyboardInterrupt:
    pass
  with open('data/model.p', 'wb') as model_file:
    pickle.dump(board.oracle.Q, model_file)

  board.oracle._print_Q_summary_snapshot()
