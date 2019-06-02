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

  def __init__(self, board):
    # Q function to be initialized randomly.
    # The domain is the tuple: (hash of the small square, action)
    self.Q = {}
    self.board = board

  def _unpack(self, small_square):
    unpacked = tuple(tuple(row) for row in small_square)
    return unpacked

  @staticmethod
  def _print_Q_state(state):
    landscape, action = state
    landscape = '\n'.join([' '.join(line) for line in landscape])
    print(landscape)
    print('Action: ', action)

  def _print_Q_summary(self):
    num_states = len(self.Q.keys())
    print('Number of states explored: ', num_states)

    num_characters = 4
    num_actions = 4
    sight_length = 3
    total_num_states = num_characters**(sight_length**2) * num_actions
    print('Number of total possible states: ', total_num_states)

    print('Ratio of states explored over total: ', num_states/total_num_states)

    try:
      # TODO: Check that snakes dying in the beginning are shorter than snakes dying in the end.
      print('Average length of dead snakes: %.2f' %
            (sum(self.board.lengths_of_dead_snakes) / len(self.board.lengths_of_dead_snakes)))
    except ZeroDivisionError:
      pass

    # Iterate over the keys of the Oracle (the states that the Oracle has seen)
    # and pick out the states associated with eating and not eating fruit.
    # State contains the action in the last slot (-1 in Python).
    eat_fruit_Q = [self.board.oracle.Q[v] for v in self.board.oracle.Q if v[1][-1] == 1]
    no_fruit_Q = [self.board.oracle.Q[v] for v in self.board.oracle.Q if v[1][-1] == 0]
    #TODO: it would be interesting to see the values of the states adjacent to the ones
    # associated with eating fruit.

    try:
      print('Average Q of eating states is %.2f' %
          (sum(eat_fruit_Q)/len(eat_fruit_Q)))
      print('Average Q of not eating states is %.2f' %
          (sum(no_fruit_Q) / len(no_fruit_Q)))
      print('Average Q of all states is %.2f' % (sum(self.board.oracle.Q.values()) / len(self.board.oracle.Q)))
    except ZeroDivisionError:
      pass

    print('Number of Q states is %i' % len(self.board.oracle.Q))


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
      max_Q_over_moves = -1000

      # This iterates over possible moves and selects the action with the max Q.
      for current_move_index, move in enumerate(moves):
        next_state = (self._unpack(small_square), move)
        try:
          next_Q = self.Q[next_state]
        except KeyError:  # If we've never actually seen this state before.
          next_Q = np.random.randint(0, 10)
          self.Q[next_state] = next_Q
        if next_Q > max_Q_over_moves:
          max_Q_over_moves = next_Q
          best_next_move_index = current_move_index

      # TODO: Play around with reward values, maybe need different numbers for better training.
      # If the best move (according to Q) leads us to eat a fruit, reward ourselves.
      if moves[best_next_move_index][-1] == 1:
        reward = 100
      else:
        reward = 0

    if last_move != (0, 0, 0):  # If this isn't the very first move for the snake.
      last_state = (self._unpack(last_small_square), last_move)

      # Q-learning update rule (https://en.wikipedia.org/wiki/Q-learning#Algorithm)
      try:
        self.Q[last_state] += alpha * (reward + gamma * max_Q_over_moves - self.Q[last_state])
      except KeyError:
        self.Q[last_state] = init_state_weight + alpha * (reward + gamma * max_Q_over_moves - init_state_weight)

    if len(moves) != 0:
      return best_next_move_index
    else:
      return
