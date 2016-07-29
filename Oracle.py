import numpy as np
from collections import defaultdict

class Oracle:
    """The Oracle is the agent that learns. Snakes "come" to the Oracle
    to consult her and they get advise on how to move on the board in
    order to eat fruit (positive reward) and avoid dying (negative reward)
    which happens if they have no moves left (all the possible directions to go
    into are occupied).

    The Oracle learns via Q-learning. The possible states are all the possible
    small squares that the snakes come with. A small square is the limited landscape
    that a snake sees around its head. The possible actions are the possible moves
    that a snake comes with -- a possible action includes the subset of the 4 directions
    in which it can turn (if the square in that direction is not occupied). So
    the Oracle must learn to guide the snakes given only the limited landscape and
    the legal moves for such a configuration.

    The cartesian product of states and actions/moves is the domain of the Q function
    which the Oracle must learn over many episodes. The range of the Q function
    is the set of possible rewards.

    From the point of view of the Oracle, this is a stochastic problem because performing
    a particular move on a particular state doesn't always lead to the same next state. This
    is because after we choose a move for a state, another snake or a fruit could have
    randomly popped into the limited landscape of the next state. """

    def __init__(self):
        # Q function initialized randomly.
        self.Q = defaultdict(lambda : 20) #np.random.uniform())

    def consult(self, small_square, moves, last_small_square, last_move):
        # Each move in moves is a triple (delta_x, delta_y, action) where
        # action tells us whether the move to (delta_x, delta_y) leads us to a fruit (if action is 1).

        alpha = 0.1 # Learning rate.
        gamma = 0.5 # Discount factor.

        if len(moves) == 0:
            # There are no moves and the snake has to die. Reward is negative.
            reward = -100
            max_Q_over_moves = 0 # There is no reward after death.
        else:
            # We need to select the next state from the possible next states (the moves).
            next_move_index = np.random.randint(0, len(moves))

            max_Q_over_moves = self.Q[(tuple(tuple(row) for row in small_square), moves[next_move_index])]
            best_next_move_index = next_move_index
            for current_move_index, move in enumerate(moves):
                next_state = (tuple(tuple(row) for row in small_square), move)
                next_Q = self.Q[next_state]
                if next_Q > max_Q_over_moves:
                    max_Q_over_moves = next_Q
                    best_next_move_index = current_move_index

            # If the best move (according to Q) leads us to eat a fruit, reward ourselves.
            if moves[best_next_move_index][-1] == 1:
                reward = 100
            else:
                reward = 0

        if last_move != (0, 0, 0):
            last_state = (tuple(tuple(row) for row in last_small_square), last_move)
            self.Q[last_state] += alpha * (reward + gamma * max_Q_over_moves - self.Q[last_state])

        if len(moves) == 0:
            return
        else:
            return best_next_move_index
