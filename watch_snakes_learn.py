import time
from Board import Board

if __name__ == '__main__':
    board = Board(30, 6)

    num_frames = 200
    delay = .05

    for _ in range(num_frames):
        board.print()
        board.tick()
        time.sleep(delay)
