import numpy as np
import sys
import re
from Oracle import Oracle


class Board:
  empty = 0
  # Indicates that a square on the board is occupied (either a wall or part of a snake).
  occupied = 255
  fruit = 240
  head = 230  # The head of the snake is special
  # Used when displaying the board (displays a symbol instead of a number).
  value_to_character = {empty: ' ', occupied: '*', fruit: '@', head: '&'}

  def __init__(self, length=30, num_snakes=1, are_snakes_helpless=False, are_snakes_learning=False):
    # `points` is a nested list representing a square with sides `length`.
    self.points = [[self.occupied]*(length + 2)]  # Create top(?) border
    self.points += [
      # border          empty space           border
      [self.occupied] + [self.empty]*length + [self.occupied] for _ in range(length)
    ]
    self.points += [[self.occupied]*(length + 2)]  # Create bottom(?) border
    self.X = len(self.points)
    self.Y = len(self.points)
    self.points = np.array(self.points)  # Give `points` numpy properties

    self.printer = Reprinter()

    self.frame = 0

    self.oracle = Oracle()  # `Oracle` is the reinforcement learning agent.

    self.are_snakes_helpless = are_snakes_helpless  # If True, makes snakes move randomly.
    self.are_snakes_learning = are_snakes_learning

    self.initial_num_snakes = num_snakes

    self.snakes = []
    for _ in range(self.initial_num_snakes):
      self.snakes.append(Snake(self))  # snakes are randomly placed.

    # For debugging: if reinforcement learning is actually working, then
    # the lengths of snakes (upon inevitable dying) should be increasing. 
    self.lengths_of_dead_snakes = []

    self.fruits = {}  # Maps points to where the fruit is located.

  def get_drawing(self):
    """Takes `points` and maps them to displayable characters"""
    points = self.points.tolist()
    for i, line in enumerate(points):
      for j, character in enumerate(line):
        points[i][j] = self.value_to_character[character]

    return points

  def is_point_empty(self, x, y):
    return self.points[x][y] == self.empty

  def is_point_fruit(self, x, y):
    if (x, y) in self.fruits:
      return True
    else:
      return False

  def find_empty_point(self):
    """Used for finding empty spaces for placing fruits and snakes. """
    while True:  # Not guaranteed to randomly find an empty space.
      x, y = np.random.randint(0, self.X), np.random.randint(0, self.Y)
      if self.is_point_empty(x, y):
        return x, y

  def tick(self):
    """This function moves each object a frame forward. This function advances time."""

    # TODO: Does this implementation detail break something?
    # Let's update the ticks of snakes in a random order (so
    # neither snake gets to go first).
    for snake in sorted(self.snakes, key=lambda x : np.random.uniform(0, 1)):
      snake.tick()

    # TODO: Does this implementation detail break something?
    # Let's update the fruit in a totally random fashion.
    if len(self.snakes) > 0:
      # We want to keep about twice as many fruit on screen as there are snakes.
      if len(self.fruits) < len(self.snakes) * 2:
        # Don't want to change things too much, so only do this action every 5 frames.
        if self.frame % 5 == 0:
          fruit = Fruit(self)  # New fruit
          self.fruits[fruit.root] = fruit
      # Otherwise, let's randomly kill fruit or plant new fruit.
      else:
        if self.frame % 10 == 0:  # Every 10 frames
          if np.random.uniform(0, 1) < 0.5:  # Killing a fruit
            i = np.random.randint(0, len(self.fruits))  # Pick a random fruit.
            fruit = self.fruits[list(self.fruits.keys())[i]]
            root = fruit.root
            fruit.die()  # Die fruit die!
            del self.fruits[root]  # Delete the key (so we don't keep tracking it)
          else:  # Planting a new fruit.
            fruit = Fruit(self)
            self.fruits[fruit.root] = fruit
    if self.initial_num_snakes > len(self.snakes):
      self.snakes.append(Snake(self))

    self.frame += 1  # Update keeping track of time.

  def print(self):
    """Calling this method will draw the current board in the console. """
    points = self.get_drawing()
    text = '\n'.join([' '.join(line) for line in points])
    self.printer.reprint(text)


class Fruit:
  body = 240

  def __init__(self, board):
    self.board = board  # Fruit's board.
    self.root = board.find_empty_point()
    board.points[self.root[0]][self.root[1]] = self.body

  def die(self):
    # Reset the fruit's root point to be empty on the board.
    self.board.points[self.root[0]][self.root[1]] = self.board.empty


class Snake:
  """Implementation detail: the snake is not the agent that learns, that's the Oracle.
  The snake is more like a puppet of the Oracle.

  What does a typical state for a snake look like?

  The following is supposed to be a 5x5 empty square
  with the snake's head in the middle. There is a fruit in the field of vision
  in the top left corner.
  =======
  |@    |
  |     |
  |  &  |
  |  *  |
  |  *  |
  =======
  There are 5x5 squares here. The snake has at most 3 actions it can take: move up, left,
  or right. Baby snakes (just the head) have 4 actions (adult snakes "can't turn backwards").

  Since each square can be either (1) empty, (2) snake head, (3) snake body, (4) occupied, or
  (5) fruit, we estimate the size of the state space to be 5^(5*5). That's a crazy number.

  #TODO: reduce the state space size (limit field of vision or the number of
  types of objects). For example, if we (somehow) reduce the number of objects to
  two types (occupied or not) and limit the the field of vision to 3x3, then
  we estimate the size of the state space to be 2^(3*3) = 512. This state space is a lot
  more learnable.
  #TODO: check if the base in the calculation is actually the number of types of states.
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
  So basically the snake's head is always in the middle of this square and things
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
  We achieve the new state by moving the head into the chosen direction, moving the body,
  recentering, and then doing a rotation. By including rotations, we can reduce the
  size of the state space by 75% (this will make it look like the snake always just
  moved up).
  # TODO: implement rotations.
  """

  body = 255  # Each piece of the snake is marked as such (except the head).
  head = 230

  def __init__(self, board):
    self.is_helpless = board.are_snakes_helpless
    self.is_learning = board.are_snakes_learning

    self.board = board  # Snake's board.
    self.dots = [board.find_empty_point()]  # Assume that head is the last point.

    #TODO: We don't have any sort of estimate for how long the training should be
    # given the approximate size of the Q-domain space.

    # The length of the small square (defines the limited landscape).
    self.sight_length = 5  # Always make it odd to be able to center the snake's head.
    for x, y in self.dots:
      board.points[x][y] = self.body

    self.oracle = board.oracle  # Every snake gets access to the same Oracle.

    # I think these fields will also be helpful when implementing the rotation
    # idea to reduce the state space by 75%.
    self.last_small_square = np.array([0])
    self.last_relative_move = (0, 0, 0)

  def get_small_square(self):
    """Gets the state around the snake's head. """
    the_head = self.dots[-1]
    return Snake._get_small_square(the_head, (self.sight_length-1)//2, self.board.points)

  @staticmethod
  def _get_small_square(position, square_radius, points):
    y, x = position

    # Recalculate the "radius" if the head is near the border.
    legal_radius = square_radius
    if x-square_radius < 0:
      legal_radius = x
    elif x+square_radius >= len(points):
      legal_radius = len(points)-x

    if y-square_radius < 0:
      legal_radius = min(legal_radius, y)
    elif y+square_radius >= len(points):
      legal_radius = min(legal_radius, len(points) - y)

    #TODO: Does this implementation detail break anything?
    # Handle borders and corners (the small square just becomes smaller).
    left = x-legal_radius
    right = x+legal_radius
    top = y-legal_radius
    bottom = y+legal_radius

    small_square = points[top:bottom+1, left:right+1].copy()

    return small_square

  def find_moves(self):
    """Find legal moves for the snake. """

    head = self.dots[-1]  # Coordinate of the head.
    moves = []  # Absolute moves.
    relative_moves = []
    # For each of the 4 possible directions...
    for delta_x, delta_y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      new_x, new_y = head[0] + delta_x, head[1] + delta_y
      if self.board.is_point_empty(new_x, new_y):
        moves.append((new_x, new_y, 0))  # 0 indicates we do not eat anything.
        relative_moves.append((delta_x, delta_y, 0))
      elif self.board.is_point_fruit(new_x, new_y):
        moves.append((new_x, new_y, 1))  # 1 indicates we are eating the fruit at this point.
        relative_moves.append((delta_x, delta_y, 1))
      # Could have another condition here eventually so leaving the elif above.
    # Note: we might not actually find any legal moves.
    # In that case, the returned lists will be empty (that snake will die die die).
    return moves, relative_moves

  def move(self, new_x, new_y, action):
    """Once a move and an action is determined, apply them. """

    # Sanity check that the move we are about to apply actually changes
    # where the head is going to be.
    assert self.dots[-1] != (new_x, new_y)

    if action == 0:  # 0 indicates that we do not eat anything.
      # We are moving the snake up...
      tail = self.dots[0]
      # Last frame's tail point becomes empty in the next frame.
      self.board.points[tail[0]][tail[1]] = self.board.empty

      # If the body is non-empty make last frame's head a body particle for the new frame.
      if len(self.dots) > 1:
        head = self.dots[-1]
        self.board.points[head[0]][head[1]] = self.body
      self.dots.append((new_x, new_y))
      self.dots = self.dots[1:] # TODO: Possibly something to optimize.
      head = self.dots[-1]
      self.board.points[head[0]][head[1]] = self.head
    elif action == 1:  # 1 indicates that we are eating a fruit.
      # We are prepared to eat the fruit at the new point.

      # TODO: this block of code is possibly buggy (leading to overgrown snakes).
      # Kill the fruit at new point.
      self.board.fruits[(new_x, new_y)].die()
      # Grow our snake into this point.
      head = self.dots[-1]
      self.board.points[head[0]][head[1]] = self.body
      self.dots.append((new_x, new_y))
      head = self.dots[-1]
      self.board.points[head[0]][head[1]] = self.head

  def tick(self):
    """Calling this method essentially advances the snake into the future. """
    moves, relative_moves = self.find_moves()

    # Helpless snakes move randomly (this was used in testing).
    # NOTE: only calling oracle.consult updates the Q-function.
    if self.is_helpless:
      if len(moves) == 0:
        self.die()
        return

      new_x, new_y, action = moves[np.random.randint(0, len(moves))]

    else: # Non-helpless snakes consult the Oracle.
      #TODO: Should the small square be rotated so that the previous
      # move always points to the top of the small square?
      small_square = self.get_small_square()  # This is the current state.

      # Gives back the Oracle's pick from the legal moves
      # or None if the snake is destined to die.
      move_index = self.oracle.consult(small_square, relative_moves,
                                       self.last_small_square, self.last_relative_move)

      if move_index is None:
        self.die()
        return

      new_x, new_y, action = moves[move_index]
      self.last_small_square = small_square  # Last state.
      self.last_relative_move = relative_moves[move_index]  # Last Q-learning action.

    self.move(new_x, new_y, action)

  def die(self):
    self.board.lengths_of_dead_snakes.append(len(self.dots))
    i = self.board.snakes.index(self)
    for dot in self.dots:
      self.board.points[dot[0]][dot[1]] = self.board.empty
    del self.board.snakes[i]


class Reprinter:
  """Credit goes to http://stackoverflow.com/a/15586020/1397712"""

  def __init__(self):
    self.text = ''

  def moveup(self, lines):
    for _ in range(lines):
      sys.stdout.write("\x1b[A")

  def reprint(self, text):
    text = '\n' + text + '\n'  # Surround any text with empty lines (added by ijkilchenko).
    # Clear previous text by overwriting non-spaces with spaces.
    self.moveup(self.text.count("\n"))
    sys.stdout.write(re.sub(r"[^\s]", " ", self.text))

    # Print new text
    lines = min(self.text.count("\n"), text.count("\n"))
    self.moveup(lines)
    sys.stdout.write(text)
    self.text = text
