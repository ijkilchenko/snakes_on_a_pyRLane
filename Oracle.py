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
    # Q function to be initialized randomly.
    # The domain is the tuple: (hash of the small square, action)
    self.Q = {}

  # TODO: Maybe this implementation detail breaks things?
  def _my_hash(self, small_square, debug=False):
    if debug:
      return tuple(tuple(row) for row in small_square)
    else:
      return small_square.tostring()

  def consult(self, small_square, moves, last_small_square, last_move):
    # TODO: This function is likely to have q-learning related bugs.
    # Each move in moves is a triple (delta_x, delta_y, action) where
    # action tells us whether the move to (delta_x, delta_y) leads us to a fruit (if action is 1).

    alpha = 1  # Learning rate.
    gamma = 0.1  # Discount factor.

    init_state_weight = 0

    if len(moves) == 0:
      # There are no moves and the snake has to die. Reward is negative.
      reward = -100
      max_Q_over_moves = 0  # There is no reward after death :)
    else:
      # We need to select the next state from the possible next states (the moves).
      next_move_index = np.random.randint(0, len(moves))

      #TODO(alex): reimplement this
      # Right now what happens is that if moves are equifavorable, the move is picked randomly.
      # What we want is we want to initialize the reward randomly.
      try:
        max_Q_over_moves = self.Q[self._my_hash(small_square), moves[next_move_index]]
      except KeyError:
        max_Q_over_moves = init_state_weight

      best_next_move_index = next_move_index

      # This iterates over possible moves and selects the action with the max Q.
      for current_move_index, move in enumerate(moves):
        next_state = (self._my_hash(small_square), move)
        try:
          next_Q = self.Q[next_state]
        except KeyError:  # If we've never actually seen this state before.
          next_Q = init_state_weight
        if next_Q > max_Q_over_moves:
          max_Q_over_moves = next_Q
          best_next_move_index = current_move_index

      # TODO: Play around with reward values, maybe need different numbers for better training.
      # If the best move (according to Q) leads us to eat a fruit, reward ourselves.
      if moves[best_next_move_index][-1] == 1:
        reward = 100
      else:
        reward = 10

    if last_move != (0, 0, 0):  # If this isn't the very first move for the snake.
      last_state = (self._my_hash(last_small_square), last_move)

      # Q-learning update rule (https://en.wikipedia.org/wiki/Q-learning#Algorithm)
      try:
        self.Q[last_state] += alpha * (reward + gamma * max_Q_over_moves - self.Q[last_state])
      except KeyError:
        self.Q[last_state] = init_state_weight + alpha * (reward + gamma * max_Q_over_moves - init_state_weight)

    if len(moves) != 0:
      return best_next_move_index
    else:
      return
