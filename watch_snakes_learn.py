import time
from Board import Board

if __name__ == '__main__':
    board = Board(30, 10, are_snakes_helpless=False)

    num_frames = 20000
    delay = .05

    for _ in range(num_frames):
        board.print()
        board.tick()
        time.sleep(delay)

    print('Average length of dead snakes: %.2f' %
          (sum(board.lengths_of_dead_snakes)/len(board.lengths_of_dead_snakes)))

    eat_fruit_Q = [board.oracle.Q[v] for v in board.oracle.Q if v[1][-1] == 1]
    no_fruit_Q = [board.oracle.Q[v] for v in board.oracle.Q if v[1][-1] == 0]

    print('Average Q of eating states is %.2f' %
          (sum(eat_fruit_Q)/len(eat_fruit_Q)))

    print('Average Q of not eating states is %.2f' %
          (sum(no_fruit_Q) / len(no_fruit_Q)))

    print('Average Q of all states is %.2f' % (sum(board.oracle.Q.values())/len(board.oracle.Q)))
