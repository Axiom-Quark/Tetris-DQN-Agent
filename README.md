This Deep Q Network utilizes Pytorch in order to train an agent on the game Tetris.

There are two main files: main.py contains just what is neccessary to train the model and is optimized for speed without altering training efficiency, main_expanded.py contains a simple but fully visualized Tetris game which I wrote in Pygame and provides more frequent and more detaied visualizations of training results.

How it works: This agent utilizes Reinforcement Learning to determine which actions should be taken in order to maximizes reward. The use of Q-Learning allows for the agent to seek future rewards instead of more immediate rewards.

Training Outcome: At first the agent selects seemingly random actions. After several hundred episodes it begins to play at an human-esque level. After several thousand episodes it is able to play at a level that surpasses what a human capable of. After the 10,000 assigneged episodes it is able to make optimal decisions nearly instuntaniously and loss has been reduced to near-zero.
