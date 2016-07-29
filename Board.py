import numpy as np
import sys
import re
from Oracle import Oracle

class Board():
    empty = 0
    occupied = 255
    fruit = 240
    head = 230
    value_to_character = {empty: ' ', occupied: '*', fruit: '@', head: '&'}

    def __init__(self, length=30, num_snakes=1, are_snakes_helpless=False):
        self.points = [[self.occupied]*(length + 2)]
        self.points += [[self.occupied] + [self.empty]*length + [self.occupied] for _ in range(length)]
        self.points += [[self.occupied]*(length + 2)]
        self.X = len(self.points)
        self.Y = len(self.points)
        self.points = np.array(self.points)

        self.printer = Reprinter()

        self.frame = 0

        self.oracle = Oracle()

        self.are_snakes_helpless = are_snakes_helpless

        self.initial_num_snakes = num_snakes

        self.snakes = []
        for _ in range(self.initial_num_snakes):
            self.snakes.append(Snake(self))

        self.lengths_of_dead_snakes = []

        self.fruits = {} # Maps points to where the fruit is located.

    def get_drawing(self):
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
        while True:
            x, y = np.random.randint(0, self.X), np.random.randint(0, self.Y)
            if self.is_point_empty(x, y):
                return (x, y)

    def tick(self):
        # Let's update the ticks of snakes in a random order (so
        # neither snake gets to go first).
        for snake in sorted(self.snakes, key=lambda x : np.random.uniform(0, 1)):
            snake.tick()

        # Let's update the fruit in a totally random fashion.

        if len(self.snakes) > 0:
            ## We want to keep about twice as many fruit on screen as there are snakes.
            if len(self.fruits) < len(self.snakes)*2:
                if self.frame % 5 == 0:
                    fruit = Fruit(self)
                    self.fruits[fruit.root] = fruit
            # Otherwise, let's randomly kill fruit or plant new fruit.
            else:
                if self.frame % 10 == 0:
                    if np.random.uniform(0, 1) < 0.5:
                        i = np.random.randint(0, len(self.fruits))
                        fruit = self.fruits[list(self.fruits.keys())[i]]
                        root = fruit.root
                        fruit.die()
                        del self.fruits[root]
                    else:
                        fruit = Fruit(self)
                        self.fruits[fruit.root] = fruit
        if self.initial_num_snakes > len(self.snakes):
            self.snakes.append(Snake(self))

        self.frame += 1

    def print(self):
        points = self.get_drawing()
        text = '\n'.join([' '.join(line) for line in points])
        self.printer.reprint(text)

class Fruit():
    body = 240

    def __init__(self, board):
        self.board = board # Fruit's board.
        self.root = board.find_empty_point()
        board.points[self.root[0]][self.root[1]] = self.body

    def die(self):
        # Reset the fruit's root point to be empty on the board.
        self.board.points[self.root[0]][self.root[1]] = self.board.empty

class Snake():
    body = 255
    head = 230

    def __init__(self, board):
        self.is_helpless = board.are_snakes_helpless

        self.board = board # Snake's board.
        self.dots = [board.find_empty_point()] # Assume that head is the last point.
        # The length of the small square (defines the limited landscape).
        # If it's set to 5, then the size of the Q-domain space is (at most) 33,554,432.
        self.sight_length = 3 # Always make it odd to center the snake's head.
        for x, y in self.dots:
            board.points[x][y] = self.body

        self.oracle = board.oracle # Every snake gets access to the same Oracle.

        self.last_small_square = np.array([0])
        self.last_relative_move = (0, 0, 0)

    def get_small_square(self):
        return Snake._get_small_square(self.dots[-1], (self.sight_length-1)//2, self.board.points)

    @staticmethod
    def _get_small_square(position, square_radius, points):
        y, x = position

        legal_radius = square_radius
        if x-square_radius < 0:
            legal_radius = x
        elif x+square_radius >= len(points):
            legal_radius = len(points)-x

        if y-square_radius < 0:
            legal_radius = min(legal_radius, y)
        elif y+square_radius >= len(points):
            legal_radius = min(legal_radius, len(points) - y)

        # Handle borders and corners (the small square just becomes smaller).
        left = x-legal_radius
        right = x+legal_radius
        top = y-legal_radius
        bottom = y+legal_radius

        small_square = points[top:bottom+1, left:right+1].copy()

        return small_square

    def find_moves(self):
        head = self.dots[-1]
        moves = [] # Absolute moves.
        relative_moves = []
        for delta_x, delta_y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = head[0] + delta_x, head[1] + delta_y
            if self.board.is_point_empty(new_x, new_y):
                moves.append((new_x, new_y, 0)) # 0 indicates we do not eat anything.
                relative_moves.append((delta_x, delta_y, 0))
            elif self.board.is_point_fruit(new_x, new_y):
                moves.append((new_x, new_y, 1)) # 1 indicates we are eating the fruit at this point.
                relative_moves.append((delta_x, delta_y, 1))
            # Could have another condition here eventually so leaving the elif above.
        return moves, relative_moves

    def move(self, new_x, new_y, action):
        assert self.dots[-1] != (new_x, new_y)
        if action == 0:
            tail = self.dots[0]
            self.board.points[tail[0]][tail[1]] = self.board.empty

            if len(self.dots) > 1:
                head = self.dots[-1]
                self.board.points[head[0]][head[1]] = self.body
            self.dots.append((new_x, new_y))
            self.dots = self.dots[1:] # Possibly something to optimize.
            head = self.dots[-1]
            self.board.points[head[0]][head[1]] = self.head
        elif action == 1:
            # We are prepared to eat the fruit at the new point.

            # Kill the fruit at new point.
            self.board.fruits[(new_x, new_y)].die()
            # Grow our snake into this point.
            head = self.dots[-1]
            self.board.points[head[0]][head[1]] = self.body
            self.dots.append((new_x, new_y))
            head = self.dots[-1]
            self.board.points[head[0]][head[1]] = self.head

    def tick(self):
        moves, relative_moves = self.find_moves()

        if self.is_helpless: # Helpless snakes move randomly (this was used in testing).
            if len(moves) == 0:
                self.die()
                return

            new_x, new_y, action = moves[np.random.randint(0, len(moves))]

        else: # Non-helpless snakes consult the Oracle.
            # TODO: Should the small square be rotated so that the previous
            # move always points to the top of the small square?
            small_square = self.get_small_square() # This is the current state.
            # Gives back the Oracle's pick from the moves
            # or None if the snake is destined to die.
            move_index = self.oracle.consult(small_square, relative_moves,
                                             self.last_small_square, self.last_relative_move)

            if move_index is None:
                self.die()
                return

            new_x, new_y, action = moves[move_index]
            self.last_small_square = small_square # Last state.
            self.last_relative_move = relative_moves[move_index] # Last Q-learning action.

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
        text = '\n' + text + '\n' # Surround any text with empty lines (added by ijk).
        # Clear previous text by overwritig non-spaces with spaces
        self.moveup(self.text.count("\n"))
        sys.stdout.write(re.sub(r"[^\s]", " ", self.text))

        # Print new text
        lines = min(self.text.count("\n"), text.count("\n"))
        self.moveup(lines)
        sys.stdout.write(text)
        self.text = text
