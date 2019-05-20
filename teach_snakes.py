import pickle
from Board import Board

if __name__ == '__main__':
    board = Board(30, 1, are_snakes_learning=True)

    num_frames = 2000000

    try:
        for _ in range(num_frames):
            board.tick()
    except KeyboardInterrupt:
        pass

    with open('models/model.p', 'wb') as model_file:
        pickle.dump(board.oracle.Q, model_file)
