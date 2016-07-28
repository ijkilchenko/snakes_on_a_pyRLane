import numpy as np
import sys
import re

class Board():
    empty = 0
    occupied = 255
    value_to_character = {empty: ' ', occupied: '*'}

    def __init__(self, length=30, num_snakes=1):
        self.points = [[self.occupied]*(length + 2)]
        self.points += [[self.occupied] + [self.empty]*length + [self.occupied] for _ in range(length)]
        self.points += [[self.occupied]*(length + 2)]
        self.X = len(self.points)
        self.Y = len(self.points)
        self.points = np.array(self.points)

        self.printer = Reprinter()

        self.frame = 0

        self.snakes = []
        for _ in range(num_snakes):
            self.snakes.append(Snake(self))

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
            # We want to keep about twice as many fruit on screen as there are snakes.
            if 2*len(self.fruits) < len(self.snakes):
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

        self.frame += 1

    def print(self):
        points = self.get_drawing()
        text = '\n'.join([' '.join(line) for line in points])
        self.printer.reprint(text)

class Fruit():
    body = 255

    def __init__(self, board):
        self.board = board # Fruit's board.
        self.root = board.find_empty_point()
        board.points[self.root[0]][self.root[1]] = self.body

    def die(self):
        # Reset the fruit's root point to be empty on the board.
        self.board.points[self.root[0]][self.root[1]] = self.board.empty

class Snake():
    body = 255
    head = 255

    def __init__(self, board):
        self.board = board # Snake's board.
        self.dots = [board.find_empty_point()] # Assume that head is the last point.
        for x, y in self.dots:
            board.points[x][y] = self.body

    def find_moves(self):
        head = self.dots[-1]
        moves = []
        for delta_x in [-1, 1]:
            for delta_y in [-1, 1]:
                new_x, new_y = head[0] + delta_x, head[1] + delta_y
                if self.board.is_point_empty(new_x, new_y):
                    moves.append((new_x, new_y, 0)) # 0 indicates we do not eat anything.
                elif self.board.is_point_fruit(new_x, new_y):
                    moves.append((new_x, new_y, 1)) # 1 indicates we are eating the fruit at this point.
        return moves

    def move(self, new_x, new_y, action):
        assert self.dots[-1] != (new_x, new_y)
        if action == 0:
            tail = self.dots[0]
            self.board.points[tail[0]][tail[1]] = self.board.empty

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
        moves = self.find_moves()

        if len(moves) == 0:
            self.die()
            return

        # This is where we would consult artificial intelligence, but
        # we just move randomly for now.
        new_x, new_y, action = moves[np.random.randint(0, len(moves))]
        self.move(new_x, new_y, action)

    def die(self):
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
