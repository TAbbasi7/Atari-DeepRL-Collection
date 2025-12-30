# 🕹️ Vanilla Deep Q-Network (DQN)
### The Foundation of Deep Reinforcement Learning

## 📖 Overview
This directory contains the implementation of the classic **DQN algorithm** as introduced by DeepMind in their 2015 Nature paper (*Human-level control through deep reinforcement learning*).

It serves as the **baseline** for this repository, demonstrating how a Convolutional Neural Network (CNN) can learn to play Atari Pong directly from raw pixels.

## ⚙️ Core Mechanisms implemented
1.  **Experience Replay Buffer:**
    * Stores transitions `(state, action, reward, next_state, done)`.
    * Randomly samples batches to break temporal correlations in data, stabilizing training.
2.  **Target Network:**
    * A frozen copy of the main network used to calculate target Q-values.
    * Updated every few thousand steps to prevent feedback loops and oscillation.
3.  **Huber Loss:**
    * Used instead of MSE (Mean Squared Error) to be more robust against outliers (gradient clipping).

## 📂 File Structure
* `1_DQN_Baseline.py`: The main script containing the Agent, ReplayBuffer, and Training Loop.

## 🚀 How to Run
```bash
python 1_DQN_Baseline.py