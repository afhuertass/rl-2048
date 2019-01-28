# A Reinforcement learning try to solve the 2048 Game
Trying to use RL to teach a neural network to play the 2048 game


This repository contains code applying the Reinforcement learning method call Asyncronus Advantage Actor Critic to train a single agent of the task of solving the puzzle game 2048

In this game the user is challenged to find the better way to move the pieces in the board in order to achieve a better score.

The code works on top of OpenAI Gym, where a custom enviroment for the game has been implemented by me. 
The core of the A3C method is the parallel exploration of the space of posible actions, the code runs indepedent agents acording to the number of avaliable cores, each agent is independent of the others in matters of environment, but they share and update the weights of the neural network

Built in top of keras 
Ongoing project. 
