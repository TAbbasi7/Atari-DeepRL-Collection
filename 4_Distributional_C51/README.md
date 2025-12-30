# 📊 Categorical DQN (C51)
### A Distributional Perspective on Reinforcement Learning

## 📖 Overview
This directory contains the implementation of **Categorical DQN (C51)**, based on the groundbreaking paper *A Distributional Perspective on Reinforcement Learning* (Bellemare et al., 2017).

Traditional RL algorithms (like DQN) model the **expected value** (mean) of the return: $Q(s, a) = \mathbb{E}[R]$.
**C51**, however, learns the **full probability distribution** of the returns. This allows the agent to distinguish between a risky action that might yield $-10$ or $+10$, and a safe action that always yields $0$, even if their means are identical.

## ⚙️ Core Mechanisms
The algorithm works by discretizing the range of possible rewards into a fixed set of support points (atoms).

### 1. The 51 Atoms
We define a support set $z$ with $N=51$ atoms, evenly spaced between $V_{min}$ and $V_{max}$:
$$z_i = V_{min} + i \Delta z$$

The network outputs a softmax probability distribution $p_i(s, a)$ over these atoms.

### 2. Projected Bellman Update
Since the Bellman update $T z_j = r + \gamma z_j$ rarely falls exactly on one of the support atoms, we project the target distribution onto the nearest neighbors in the support set. This is known as the **Categorical Projection**.

### 3. Loss Function
Instead of Mean Squared Error (MSE), we minimize the **Cross-Entropy** (or Kullback-Leibler Divergence) between the predicted distribution and the projected target distribution:
$$\mathcal{L} = - \sum_i m_i \log p_i(s, a)$$

## 📂 Code Implementation
The logic is implemented in `4_Distributional_C51.py`.
* **Hyperparameters:** `atom_num=51`, `min_value=-10`, `max_value=10`.
* **Network Output:** The final layer has `action_dim * atom_num` outputs, reshaped to `(actions, 51)`.

## 🛠️ Dependencies
```bash
pip install gymnasium[atari] ale-py opencv-python tensorflow shimmy