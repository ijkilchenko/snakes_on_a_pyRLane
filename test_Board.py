import unittest
import time
import numpy as np
from Board import Board, Snake

class TestBoard(unittest.TestCase):

    def test_get_drawing(self):
        board = Board(15, 0) # Make a 15x15 board.
        drawing = board.get_drawing()
        self.assertEqual(len(drawing), 17)
        self.assertEqual(len(drawing[0]), 17)
        board.print() # Prints board to stdout.

        board = Board(30, 0) # Make a 30x30 board.
        drawing = board.get_drawing()
        self.assertEqual(len(drawing), 32)
        self.assertEqual(len(drawing[0]), 32)
        board.print() # Prints board to stdout.

    def test_print_multiple_times(self):
        board = Board(15, 0) # Make a 15x15 board.
        board.print()
        time.sleep(.2)
        board.print()
        time.sleep(.2)
        board.print()
        time.sleep(.2)
        board.print()

    def test_tick(self):
        board = Board(15, 0) # Make a 15x15 board.
        board.tick() # Moves from frame 0 to frame 1.
        board.tick() # Moves from frame 1 to frame 2.
        board.tick() # Moves from frame 2 to frame 3.
        self.assertEqual(board.frame, 3)

    def test_add_snake(self):
        board = Board(15, 1) # Board with 1 snake.
        board.print()

        board = Board(15, 2)  # Board with 1 snake.
        board.print()

    def test_tick_snake(self):
        board = Board(15, 2) # Make a 15x15 board.
        snake_start = board.snakes[0].dots[0]
        board.tick()
        snake_end = board.snakes[0].dots[0]
        self.assertNotEqual(snake_start, snake_end)

class TestSnake(unittest.TestCase):
    def test_get_small_square(self):
        A = np.array([[1, 1, 1, 2, 2],
                      [1, 2, 3, 5, 5],
                      [4, 5, 6, 7, 8],
                      [1, 5, 6, 8, 9],
                      [4, 4, 2, 1, 1]])

        small_square_1 = Snake._get_small_square((2,2), 2, A)
        self.assertTrue((small_square_1 == A).all())

        small_square_2 = Snake._get_small_square((1, 1), 2, A)
        expected_small_square_2 = np.array([[1, 1, 1, 2],
                                            [1, 2, 3, 5],
                                            [4, 5, 6, 7],
                                            [1, 5, 6, 8]])
        self.assertTrue((small_square_2 == expected_small_square_2).all())

if __name__ == '__main__':
    unittest.main()
