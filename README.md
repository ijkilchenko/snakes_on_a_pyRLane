# snakes_on_a_pyRLane (Work in progress)
Watch snakes on a plane learn to eat fruit and avoid each other using Reinforcement Learning (Q-learning).

## Project Structure
We have a `Board.py` file which holds the implementation for the `Board`, `Fruit`, and `Snake` (and also a `Reprinter` which helps us display the game within a console). 

Read their classes to figure out the exact implementation. Overall, the idea is that a Board starts out with some Fruits and some Snakes (exact number is changeable). Both are placed randomly on the board (but there are usually twice as many fruits as there snakes). For some period of time, snakes begin to wander around aimlessly, but they are recording how their chosen actions affect the reward (of eating a fruit and thus growing in size). At some point, the regime switches over to be not aimless but smart. At that point, each snakes consults the Oracle which is the actual reinforcement learning agent. 

There are two final scripts:
* `teach_snakes.py`
* `watch_snakes.py`

## Things to try out to debug RL
* Make initialization of the Q-function smarter with some heuristics 
