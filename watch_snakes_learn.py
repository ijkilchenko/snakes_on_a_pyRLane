import time
from Board import Board

if __name__ == '__main__':
    board = Board(30, 10, are_snakes_helpless=False)

    num_frames = 2000
    delay = .07

    for _ in range(num_frames):
        board.print()
        board.tick()
        time.sleep(delay)

    print('The average length of a dead snake was %.2f' %
          (sum(board.lengths_of_dead_snakes)/len(board.lengths_of_dead_snakes)))
