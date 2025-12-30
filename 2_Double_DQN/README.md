# ⚖️ Double Deep Q-Network (DDQN)
### Reducing Overestimation Bias in Deep Reinforcement Learning

## 📖 Overview
This directory contains the implementation of **Double DQN** (DDQN), based on the paper *Deep Reinforcement Learning with Double Q-learning* (Van Hasselt et al., 2016).

While the original DQN was a breakthrough, it suffered from a significant issue known as **Maximization Bias** (or Overestimation Bias). This algorithm mitigates that problem by decoupling the action selection from the action evaluation.

## 📉 The Problem: Overestimation Bias
In standard DQN, the target value is calculated as:
$$Y_t^{DQN} = r + \gamma \max_{a} Q(s', a; \theta_{target})$$

The `max` operator uses the same values to both **select** and **evaluate** an action. If the Q-values contain any noise, the `max` operator tends to select overestimated values, leading to unstable training and suboptimal policies.

## ⚙️ The Solution: Decoupled Updates
Double DQN solves this by using **two separate networks** for the calculation:

1.  **Selection (Online Network):** The main network $\theta$ decides *which* action is the best for the next state.
2.  **Evaluation (Target Network):** The target network $\theta'$ estimates the *value* of that chosen action.

### Mathematical Formula
$$Y_t^{DoubleDQN} = r + \gamma Q(s', \underbrace{\arg\max_{a} Q(s', a; \theta)}_{\text{Selection}}, \theta')$$

This simple change stabilizes the Q-values and often leads to higher scores in Atari games compared to vanilla DQN.

## 📂 Code Implementation
The implementation is contained in `2_Double_DQN.py`.
* **Environment:** `PongNoFrameskip-v4`
* **Framework:** TensorFlow 2.x
* **Key Logic:** inside the `_tderror_func` method, you will see the split logic for calculating `b_q_`.

## 🛠️ Dependencies
Ensure you have the following installed:
```bash
pip install gymnasium[atari] ale-py opencv-python tensorflow shimmy