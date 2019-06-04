import numpy as np
import Board
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
  is all real numbers which represent the value at that point (being in a specific state
  and taking some specific action). Note that the range it's simply the set of
  possible rewards.

  From the point of view of the Oracle, this is a stochastic problem because performing
  a particular move on a particular state doesn't always lead to the same next state. This
  is because after we choose a move for a state, another snake or a fruit could have
  randomly popped into the limited landscape of the next state.
  """

  init_state_weight = lambda _ : np.random.normal()

  reward_at_death = -10  # There is no reward after death :)
  reward_when_eating_fruit = 100
  reward_for_staying_alive = 0

  alpha = 0.1  # Learning rate
  gamma = 0.2  # Discount factor

  def __init__(self, board):
    self.board = board
    # Q function to be initialized randomly.
    # The domain type is the tuple: (small square, action)
    # The range is all real numbers
    self.Q = {}

    # T is the map of transitions (from state (landscape, move) to a map from new landscape to the count of times
    # we arrive at the new landscape -- almost a probability distribution but not normalized)
    self.T = defaultdict(lambda *args : defaultdict(lambda *bargs : 0))

    # Tinv is the map from landscapes to the set of states that led to that landscape at some point.
    self.Tinv = defaultdict(lambda : set())

    # F is the map from state (landscape, move) to the count of visits of that state
    self.F = defaultdict(lambda *args : 0)

  def _unpack(self, small_square):
    unpacked = tuple(tuple(row) for row in small_square)
    return unpacked

  @staticmethod
  def _print_Q_state(state):
    landscape, action = state
    landscape = '\n'.join([' '.join(line) for line in landscape])
    print(landscape)
    print('Action: ', action)

  def _print_Q_summary_snapshot(self):
    num_states, reached_states_ratio, avg_length_of_dead_snakes = self._get_Q_summary_snapshot()

    print('Num of states explored:\t', num_states)

    print('Ratio of reached states:\t', reached_states_ratio)

    print('Average length of dead snakes: %.2f' % avg_length_of_dead_snakes)

  def _get_Q_summary_snapshot(self):
    num_states = len(self.Q.keys())

    num_characters = 4  # empty, occupied, snake head, snake body
    num_actions = 4  # up, down, left, right
    landscape_length = Board.Snake.landscape_length
    total_num_states = num_characters ** (landscape_length ** 2) * num_actions

    reached_states_ratio = num_states / total_num_states

    try:
      avg_length_of_dead_snakes = sum(self.board.lengths_of_dead_snakes) / len(self.board.lengths_of_dead_snakes)
    except ZeroDivisionError:
      avg_length_of_dead_snakes = 0

    # Iterate over the keys of the Oracle (the states that the Oracle has seen)
    # and pick out the states associated with eating and not eating fruit.
    # State contains the action in the last slot (-1 in Python).
    eat_fruit_Q = [self.Q[v] for v in self.Q if v[1][-1] == 1]
    no_fruit_Q = [self.Q[v] for v in self.Q if v[1][-1] == 0]
    # TODO: it would be interesting to see the values of the states adjacent to the ones
    # associated with eating fruit.

    try:
      print('Average Q of eating states is %.2f' %
            (sum(eat_fruit_Q) / len(eat_fruit_Q)))
      print('Average Q of not eating states is %.2f' %
            (sum(no_fruit_Q) / len(no_fruit_Q)))
      print('Average Q of all states is %.2f' % (sum(self.Q.values()) / len(self.Q)))
    except ZeroDivisionError:
      pass

    return num_states, reached_states_ratio, avg_length_of_dead_snakes

  def consult(self, small_square, moves, last_small_square, last_move):
    # TODO: This function is likely to have q-learning related bugs.
    # Each move in moves is a triple (delta_x, delta_y, action) where
    # action tells us whether the move to (delta_x, delta_y) leads us to a fruit (if action is 1).

    # TODO: We don't have to initialize these here
    max_Q_val_over_moves = 0  # TODO: Only need this here in case len(moves) == 0
    best_next_move_index = 0

    small_square_unpacked = self._unpack(small_square)

    if len(moves) == 0:
      # There are no moves and the snake has to die. Reward is negative.
      reward = self.reward_at_death
    else:
      max_Q_val_over_moves = -np.inf

      # This iterates over possible moves and selects the action with the max Q.
      for current_move_index, move in enumerate(moves):
        next_state = (small_square_unpacked, move)

        try:
          next_Q_val = self.Q[next_state]
        except KeyError:  # If we've never actually seen this state before.
          next_Q_val = self.init_state_weight()
          self.Q[next_state] = next_Q_val

        if next_Q_val > max_Q_val_over_moves:
          max_Q_val_over_moves = next_Q_val
          best_next_move_index = current_move_index

      # TODO: Play around with reward values, maybe need different numbers for better training.
      # If the best move (according to Q) leads us to eat a fruit, reward ourselves.
      if moves[best_next_move_index][-1] == 1:
        reward = self.reward_when_eating_fruit
      else:
        reward = self.reward_for_staying_alive

    # TODO: What do we do when this is the last move for the snake?
    if last_move != (0, 0, 0):  # If this isn't the very first move for the snake.
      last_state = (self._unpack(last_small_square), last_move)

      # Q-learning update rule (https://en.wikipedia.org/wiki/Q-learning#Algorithm)
      try:
        self.Q[last_state] += self.alpha * (reward + self.gamma * max_Q_val_over_moves - self.Q[last_state])
      except KeyError:
        self.Q[last_state] = self.init_state_weight() + self.alpha * (reward + self.gamma * max_Q_val_over_moves - self.init_state_weight())

      self.T[last_state][small_square_unpacked] += 1
      self.Tinv[small_square_unpacked].add(last_state)
    self.F[small_square_unpacked] += 1

    if len(moves) != 0:
      return best_next_move_index
    else:
      return
