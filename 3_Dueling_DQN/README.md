# ⚔️ Dueling Network Architecture
### Efficient State Evaluation in Deep RL

## 📖 Overview
This directory hosts the implementation of the **Dueling DQN** architecture, based on the paper *Dueling Network Architectures for Deep Reinforcement Learning* (Wang et al., 2016).

Unlike Double DQN which changes the *update rule*, Dueling DQN changes the **neural network structure** itself. It explicitly separates the estimation of the state's value from the advantages of specific actions.

## 🧠 The Core Insight
In many reinforcement learning environments, it is unnecessary to know the value of each action at every timestep.
* **Example:** In *Pong*, if the ball is far away, it doesn't matter whether the paddle moves UP or DOWN. The value of the state is largely determined by the game score, not the immediate action.
* **Standard DQN:** Struggles here because it tries to learn the Q-value for every action independently.
* **Dueling DQN:** Learns a general state-value $V(s)$ which is shared across all actions, leading to faster convergence.

## ⚙️ Architecture & Math
The network features two separate streams of fully connected layers after the convolutional feature extractor:

1.  **Value Stream $V(s; \theta, \beta)$:** Outputs a single scalar (how good is the current state?).
2.  **Advantage Stream $A(s, a; \theta, \alpha)$:** Outputs a vector of size $|A|$ (how much better is action $a$ compared to the average?).

### The Aggregation Layer
To combine these streams back into Q-values, we use the **Mean Aggregation** method to ensure identifiability:

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$$

This forces the Advantage function to have zero mean, stabilizing the optimization.

## 📂 Code Implementation
The logic is implemented in `3_Dueling_DQN.py`.
* **Framework:** TensorFlow 2 / Keras
* **Model Class:** `DuelingQFunc` (inherits from `tf.keras.Model`)
* **Streams:** You will see separate `self.fc_value` and `self.fc_advantage` layers in the code.

## 🛠️ Dependencies
```bash
pip install gymnasium[atari] ale-py opencv-python tensorflow shimmy