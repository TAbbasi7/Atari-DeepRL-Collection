<div align="center">

# 🎮 Atari Deep Reinforcement Learning Zoo
### TensorFlow 2 Implementation | PongNoFrameskip-v4

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Demo_Ready-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

<p align="center">
  <b>A step-by-step evolution of Deep Q-Networks (DQN)</b><br>
  From the original 2015 Baseline to Distributional C51 Agents.
</p>

</div>

---

## 📖 Project Overview
This repository contains a comprehensive collection of Deep Reinforcement Learning algorithms implemented from scratch. The goal is to demonstrate the **progression of RL agents** on the classic Atari game **Pong**.

Each script is self-contained, heavily commented, and pre-configured in **"Demo Mode"** for rapid execution on Google Colab (T4 GPU).

### 🚀 Key Features
* **Zero-Config Demo:** All hyperparameters are tuned for a 2,000-step verification run.
* **Real-time Logging:** Custom loggers provide immediate feedback on Loss, Reward, and Epsilon.
* **Clean Architecture:** Modular code separating the *Agent*, *Replay Buffer*, and *Environment Wrappers*.

---

## 📂 Repository Structure

| Algorithm Module | Algorithm | Key Innovation |
| :--- | :--- | :--- |
| [📂 **1_DQN_Baseline**](./1_DQN_Baseline) | **Vanilla DQN** | Experience Replay & Target Networks (Nature 2015). |
| [📂 **2_Double_DQN**](./2_Double_DQN) | **Double DQN** | Decouples action selection from evaluation to fix overestimation. |
| [📂 **3_Dueling_DQN**](./3_Dueling_DQN) | **Dueling DQN** | Splits network into Value $V(s)$ and Advantage $A(s,a)$ streams. |
| [📂 **4_Distributional_C51**](./4_Distributional_C51) | **C51 (Categorical)** | Learns the full *probability distribution* of returns (51 Atoms). |
---

## 🛠️ Installation & Usage

### 1. Install Dependencies
Run the following command to install the required libraries (Gymnasium, ALE, TensorFlow):
```bash
pip install gymnasium[atari] ale-py opencv-python tensorflow shimmy