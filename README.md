# snakes_on_a_pyRLane (work in progress)
Watch snakes on a plane learn to eat fruit and avoid each other using Reinforcement Learning (Q-learning).

## Setup
We use [pynput](https://pypi.org/project/pynput/) to handle keyboard controls for rewinding, pausing/unpausing, and fastforwarding. This library doesn't require root privileges.

`pip install -r requirements.txt` to install all required modules.

## Project structure
We have a `Board.py` file which holds the implementation for the `Board`, `Fruit`, and `Snake` (and also a `Reprinter` which helps us display the game within a console). 

Read their classes to figure out the exact implementation. Overall, the idea is that a Board starts out with some Fruits and some Snakes (exact number is changeable). Both are placed randomly on the board (but there are usually twice as many fruits as there are snakes). Snakes begin to wander around, consulting the Oracle (the reinforcement learning agent), and try to maximize the reward (of eating a fruit and thus growing in size).  

## How to run
There are two final scripts:
* `teach_snakes.py` trains the snakes without displaying the game.
* `watch_snakes.py` trains the snakes while displaying the game and allows the user to control the playback. 

Each of the above scripts uses the same model file `models/model.p` (it's created if it doesn't exist). 

You can do `python watch_snakes.py` right away without training just to see them move around or
do `python teach_snakes.py` and either wait until training is done (`num_frames` is exhausted) or do
Ctrl+C (KeyboardInterrupt) and this will trigger the current model to be pickled/saved. 

To control the playback use `j`, `k`, and `l` to rewind, pause/unpause, and fastforward, respectively.  

Do `python test_Board.py` to run all the available tests. 

## Things to try out to debug RL
* Make initialization of the Q-function smarter with some heuristics 
