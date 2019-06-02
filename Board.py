import numpy as np
import re
import sys
from Oracle import Oracle


class Board:
  symbol_empty = ' '
  symbol_occupied = '*'  # Point marked as such is either a snake symbol_fruit or a board wall

  def __init__(self, side_length=30, num_snakes=10, are_snakes_random=False, are_snakes_learning=False):
    """Constructor for `Board`

    Args:
      side_length (int): length of the side of the Board
      num_snakes (int): number of snakes on the Board
      are_snakes_random (bool): If True, makes snakes move randomly
      are_snakes_learning (bool): If True, the snakes consult the Oracle
    """

    # `points` is a nested list representing a square with sides `side_length`
    self.points = [[self.symbol_occupied] * (side_length + 2)]  # Create top border
    self.points += [
      # border                   symbol_empty space                  border
      [self.symbol_occupied] + [self.symbol_empty] * side_length + [self.symbol_occupied] for _ in range(side_length)
    ]
    self.points += [[self.symbol_occupied] * (side_length + 2)]  # Create bottom border

    self.X = len(self.points)
    self.Y = len(self.points)
    self.points = np.array(self.points)  # Give `points` numpy properties

    self.frame = 0  # Keeps track of the number of frames passed

    self.oracle = Oracle(self)  # `Oracle` is the reinforcement learning agent

    self.are_snakes_random = are_snakes_random
    self.are_snakes_learning = are_snakes_learning

    self.initial_num_snakes = num_snakes
    self.snakes = []
    for _ in range(self.initial_num_snakes):
      self.snakes.append(Snake(self))  # snakes are randomly placed.

    # For debugging: if reinforcement learning is actually working, then
    # the lengths of snakes (upon inevitable dying) should be increasing.
    self.lengths_of_dead_snakes = []

    self.fruits = {}  # Maps points to where the fruit is located.

    self.printer = Reprinter()

  def get_drawing(self):
    """Gets the drawing of the way the Board looks like right now
    """

    return self.points.copy()

  def is_point_empty(self, x, y):
    return self.points[x][y] == self.symbol_empty

  def is_point_fruit(self, x, y):
    return self.points[x][y] == Fruit.symbol_fruit

  def find_empty_point(self):
    """Used for finding empty spaces for placing fruits and snakes
    """

    while True:  # Not guaranteed to randomly find an empty space
      x, y = np.random.randint(0, self.X), np.random.randint(0, self.Y)
      if self.is_point_empty(x, y):
        return x, y

  def _self_check(self):
    """We can check if the instance variables for our snakes and fruits are drawn on the Board properly
    """

    # Check that all snakes are drawn properly
    for snake in self.snakes:
      for dot in snake.dots[:-1]:
        try:
          assert self.points[dot[0]][dot[1]] == Snake.symbol_body
        except AssertionError as e:
          self.print()
          raise e
      try:
        head = snake.dots[-1]
        assert self.points[head[0]][head[1]] == Snake.symbol_head
      except AssertionError as e:
        self.print()
        raise e

    # Check that all fruits are drawn properly
    for fruit in self.fruits:
      try:
        assert self.points[fruit[0]][fruit[1]] == Fruit.symbol_fruit
      except AssertionError as e:
        self.print()
        raise e

  def tick(self):
    """This function moves each object a frame forward. This function advances time.
    """

    # TODO: Does this implementation detail break something?
    # Let's update the ticks of snakes in a random order (so no one snake always moves first)
    for snake in sorted(self.snakes, key=lambda x: np.random.uniform(0, 1)):
      snake.tick()
      self._self_check()

    # TODO: Does this implementation detail break something?
    # Let's update the fruit in a totally random fashion
    if len(self.snakes) > 0:
      # We want to keep about twice as many fruit on the screen as there are snakes
      if len(self.fruits) < len(self.snakes) * 2:
        # Don't want to change things too much, so only do this action every 5 frames
        if self.frame % 5 == 0:
          Fruit(self)  # New fruit
          self._self_check()
      # Otherwise, let's randomly kill fruit or plant new fruit
      else:
        self._self_check()
        if self.frame % 10 == 0:  # Every 10 frames
          if np.random.uniform(0, 1) < 0.5:  # Killing a fruit
            self._self_check()
            i = np.random.randint(0, len(self.fruits))  # Pick a random fruit.
            fruit = self.fruits[list(self.fruits.keys())[i]]
            fruit.die()  # Die fruit die!
            self._self_check()
          else:  # Planting a new fruit.
            Fruit(self)
            self._self_check()
    self._self_check()
    if self.initial_num_snakes > len(self.snakes):
      self.snakes.append(Snake(self))
    self._self_check()

    self.frame += 1  # Update keeping track of time.

  def print(self):
    """Calling this method will draw the current board in the console
    """

    drawing = self.get_drawing()
    drawing = '\n'.join([' '.join(line) for line in drawing])
    print(drawing)

  def reprint(self):
    """Calling this method will overwrite last printed board in the console
    """

    drawing = self.get_drawing()
    drawing = '\n'.join([' '.join(line) for line in drawing])
    self.printer.reprint(drawing)

  def print_drawing(self, drawing):
    """Calling this method will draw the current board in the console
    """

    text = 'Use j, k, l to control playback\n'
    text += '\n'.join([' '.join(line) for line in drawing])
    self.printer.reprint(text)


class Fruit:
  symbol_fruit = '@'

  def __init__(self, board):
    self.root = board.find_empty_point()
    self.board = board  # Fruit's board

    board.points[self.root[0]][self.root[1]] = self.symbol_fruit
    board.fruits[self.root] = self

  def die(self):
    assert self.board.points[self.root[0]][self.root[1]] == self.symbol_fruit

    self.board.points[self.root[0]][self.root[1]] = self.board.symbol_empty
    del self.board.fruits[self.root]  # Delete the key (so we don't keep tracking it)


class Snake:
  # TODO: clean up all the _self_check()'s
  """The snake is not the agent that learns, that's the Oracle.
  The snake is more like a puppet of the Oracle.

  What does a typical state for a snake look like?

  The following is supposed to be a 5x5 empty square
  with the snake's symbol_head in the middle. There is a fruit in the field of vision
  in the top left corner.
  =======
  |@    |
  |     |
  |  &  |
  |  *  |
  |  *  |
  =======
  There are 5x5 squares here. The snake has at most 3 actions it can take: move up, left,
  or right. Baby snakes (just the symbol_head) have 4 actions (adult snakes "can't turn backwards").

  Since each square can be either (1) symbol_empty, (2) snake symbol_head, (3) symbol_occupied, or
  (4) fruit, we estimate the size of the landscape space to be 4^(5*5). That's a crazy number.
  Together with the action space of 4, that's 4*4^(5*5).

  #TODO: reduce the state space size (limit field of vision or the number of
  types of objects). For example, if we (somehow) reduce the number of objects to
  two types (occupied or not) and limit the the field of vision to 3x3, then
  we estimate the size of the state space to be 2^(3*3) = 512. This state space is a lot
  more learnable.
  Another way to reduce the state space would be to introduce some kind of similarity
  functions that collapse some states together.
  Another way (C.C.'s idea) is to append the state with more directly useful information
  such as direction to nearest fruit.

  Suppose the snake decides to move up (and it's length is 3).
  =======
  |     |
  |@    |
  |  &  |
  |  *  |
  |  *  |
  =======
  So basically the snake's symbol_head is always in the middle of this square and things
  around move instead.

  The overall idea is to learn that if there is a fruit in the field of vision, the
  snake should take smart moves that bring it closer to the future reward.

  Now let's say that the snake turns left:
  =======
  |     |
  |   @ |
  |  &  |
  | **  |
  |     |
  =======
  We can achieve the new state by moving the symbol_head into the chosen direction, moving the symbol_fruit,
  recentering, and then doing a rotation. By including rotations, we can reduce the
  size of the state space by 75% (this will make it look like the snake always just
  moved up).
  # TODO: implement these rotations.
  """

  symbol_body = '*'  # Each piece of the snake is marked as such (except the head)
  symbol_head = '&'

  landscape_length = 3  # Always make it odd to be able to center the snake's head

  def __init__(self, board):
    self.is_random = board.are_snakes_random
    self.is_learning = board.are_snakes_learning

    self.board = board  # Snake's board
    self.dots = [board.find_empty_point()]  # Assume that head is the last point

    # The length of the small square (defines the limited landscape)
    for x, y in self.dots:
      board.points[x][y] = self.symbol_head

    self.oracle = board.oracle  # Every snake gets access to the same Oracle

    # I think these fields will also be helpful when implementing the rotation
    # idea to reduce the state space by 75%
    self.last_small_square = np.array([0])
    self.last_relative_move = (0, 0, 0)

  def get_small_square(self):
    """Gets the state around the snake's head
    """

    head = self.dots[-1]
    return Snake._get_small_square(head, (self.landscape_length - 1) // 2, self.board.points)

  @staticmethod
  def _get_small_square(position, landscape_length, points):
    y, x = position

    # Recalculate the landscape_length if the head is near the border.
    legal_landscape_length = landscape_length
    if x - landscape_length < 0:
      legal_landscape_length = x
    elif x + landscape_length >= len(points):
      legal_landscape_length = len(points) - x

    if y - landscape_length < 0:
      legal_landscape_length = min(legal_landscape_length, y)
    elif y + landscape_length >= len(points):
      legal_landscape_length = min(legal_landscape_length, len(points) - y)

    # TODO: Does this implementation detail break anything?
    # Handle borders and corners (the small square just becomes smaller).
    left = x - legal_landscape_length
    right = x + legal_landscape_length
    top = y - legal_landscape_length
    bottom = y + legal_landscape_length

    small_square = points[top:bottom + 1, left:right + 1].copy()

    return small_square

  def find_moves(self):
    """Find legal moves for the snake
    """

    head = self.dots[-1]  # Coordinate of the symbol_head.
    moves = []  # Absolute moves.
    relative_moves = []

    # For each of the 4 possible directions...
    for delta_x, delta_y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      new_x, new_y = head[0] + delta_x, head[1] + delta_y

      if self.board.is_point_empty(new_x, new_y):
        moves.append((new_x, new_y, 0))  # 0 indicates we do not eat anything
        relative_moves.append((delta_x, delta_y, 0))
      elif self.board.is_point_fruit(new_x, new_y):
        moves.append((new_x, new_y, 1))  # 1 indicates we are eating the fruit at this point
        relative_moves.append((delta_x, delta_y, 1))
      # Could have another condition here eventually so leaving the elif above.

    # Note: we might not actually find any legal moves.
    # In that case, the returned lists will be empty (that snake will die die die).
    return moves, relative_moves

  def move(self, new_x, new_y, action):
    """Once a move and an action is determined, apply them.
    """

    # Sanity check (check that the new position the snake will move into does not make it intersect itself)
    try:
      assert all([self.dots[i] != (new_x, new_y) for i in range(len(self.dots))])
    except AssertionError as e:
      self.board.print()
      raise e

    self.board._self_check()

    if action == 0:  # 0 indicates that we do not eat anything.
      # We are moving the snake up...
      tail = self.dots[0]
      # Last frame's tail point becomes symbol_empty in the next frame.
      self.board.points[tail[0]][tail[1]] = self.board.symbol_empty  # Update tail to symbol_empty

      if len(self.dots) > 1:
        head = self.dots[-1]
        self.board.points[head[0]][head[1]] = self.symbol_body

      self.dots.append((new_x, new_y))
      self.dots = self.dots[1:]  # TODO: Possibly something to optimize.
      head = self.dots[-1]
      self.board.points[head[0]][head[1]] = self.symbol_head  # Update symbol_head on board
      self.board._self_check()
    elif action == 1:  # 1 indicates that we are eating a fruit.
      # We are prepared to eat the fruit at the new point.
      length_before_eating = len(self.dots)

      # Kill the fruit at new point.
      self.board.fruits[(new_x, new_y)].die()

      # Grow our snake into this point.
      head = self.dots[-1]
      self.board._self_check()
      self.board.points[head[0]][head[1]] = self.symbol_body
      self.dots.append((new_x, new_y))
      head = self.dots[-1]
      self.board.points[head[0]][head[1]] = self.symbol_head
      self.board._self_check()

      length_after_eating = len(self.dots)

      assert length_after_eating > length_before_eating
      self.board._self_check()

  def tick(self):
    """Calling this method essentially advances the snake into the future.
    """

    moves, relative_moves = self.find_moves()

    self.board._self_check()
    # Random snakes move randomly (this was used in testing).
    # NOTE: only calling oracle.consult updates the Q-function.
    if self.is_random:
      if len(moves) == 0:
        self.die()
        return

      new_x, new_y, action = moves[np.random.randint(0, len(moves))]

    else:  # Non-helpless snakes consult the Oracle.
      # TODO: Should the small square be rotated so that the previous
      # move always points to the top of the small square?
      small_square = self.get_small_square()  # This is the current landscape

      # Gives back the Oracle's pick from the legal moves
      # or None if the snake is destined to die.
      move_index = self.oracle.consult(small_square, relative_moves,
                                       self.last_small_square, self.last_relative_move)

      if move_index is None:
        self.die()
        return

      new_x, new_y, action = moves[move_index]
      self.last_small_square = small_square  # Last state
      self.last_relative_move = relative_moves[move_index]  # Last Q-learning action

    self.board._self_check()
    self.move(new_x, new_y, action)
    self.board._self_check()

  def die(self):
    self.board.lengths_of_dead_snakes.append(len(self.dots))

    for dot in self.dots:
      self.board.points[dot[0]][dot[1]] = self.board.symbol_empty

    i = self.board.snakes.index(self)
    del self.board.snakes[i]


class Reprinter:
  """Credit goes to http://stackoverflow.com/a/15586020/1397712"""

  def __init__(self):
    self.text = ''

  def moveup(self, lines):
    for _ in range(lines):
      sys.stdout.write("\x1b[A")

  def reprint(self, text):
    text = '\n' + text + '\n'  # Surround any text with symbol_empty lines (added by ijkilchenko).
    # Clear previous text by overwriting non-spaces with spaces.
    self.moveup(self.text.count("\n"))
    sys.stdout.write(re.sub(r"[^\s]", " ", self.text))

    # Print new text
    lines = min(self.text.count("\n"), text.count("\n"))
    self.moveup(lines)
    sys.stdout.write(text)
    self.text = text
