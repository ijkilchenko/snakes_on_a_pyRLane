import numpy as np

class Oracle:
    """The Oracle is the agent that learns. Snakes "come" to the Oracle
    to consult her and they get advise on how to move on the board in
    order to eat fruit (positive reward) and avoid dying (negative reward)
    which happens if they have no moves left (all the possible directions to go
    into are occupied).

    The Oracle learns via Q-learning. The possible states are all the possible
    small squares that the snakes come with. A small square is the limited landscape
    that a snake sees around its head. The possible actions are the possible moves
    that a snake comes with -- a possible action includes the subset of the 3 directions
    in which it can turn (if the square in that direction is not occupied). So
    the Oracle must learn to guide the snakes given only the limited landscape and
    the legal moves for such a configuration.

    The cartesian product of states and actions/moves is the domain of the Q function
    which the Oracle must learn over many episodes. The range of the Q function
    is the set of possible rewards. """

    def __init__(self):
        self.Q = {} # Q function.

    def consult(self, small_square, moves):
        # Each move in moves is a triple (delta_x, delta_y, action) where
        # action tells us whether moves to (delta_x, delta_y) leads us to a fruit (if action is 1).

        if len(moves) == 0:
            self.die()
            return

        return np.random.randint(0, len(moves))
