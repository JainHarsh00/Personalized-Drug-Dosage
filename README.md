# 💊 Personalized Drug Dosage using Deep Reinforcement Learning

Can we treat **medication dosage** as a **Reinforcement Learning (RL)** problem?

In standard medical treatments, drug dosage is often **fixed or rule-based**, despite the fact that every patient responds differently.  
This project explores a **personalized, adaptive dosing strategy** using **Deep Reinforcement Learning (DRL)**, where the human body is modeled as an *environment* and the medication plan is learned as a *policy*.

---

## 🧠 Core Idea

> **Find the “Goldilocks” zone** — a dosage that is **effective enough to maximize recovery** while **low enough to minimize toxicity**.

The problem is framed as a **continuous control task**, where an agent must dynamically adjust drug dosage based on patient state over time.

---

## 🔬 Problem Formulation (RL View)

- **Environment**: Human body (simulated)
- **State**: Patient health indicators (e.g. drug concentration, recovery level, toxicity)
- **Action**: Drug dosage (continuous)
- **Reward**:
  - Positive reward for recovery
  - Penalty for toxicity or overdosing
- **Goal**: Learn a safe and effective dosing policy over time

---

## 🛠 Tech Stack

### Environment
- Custom **Gymnasium** environment
- Simulates patient response to varying drug dosages
- Captures delayed effects and toxicity trade-offs

### Algorithms
- **DQN (Deep Q-Network)**  
  - Used as a baseline
  - Limited due to discrete action assumptions
- **PPO (Proximal Policy Optimization)**  
  - Handles continuous action space
  - More stable and sample-efficient

### Libraries
- Python
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib

---

## 📊 Results & Observations

- **PPO successfully learned a dynamic dosing strategy**
- Outperformed:
  - Fixed/static dosage protocols
  - DQN-based approaches in continuous settings
- Learned behavior:
  - Higher dose when recovery is slow
  - Lower dose when toxicity risk increases
- Demonstrates how RL can **adapt treatment over time** instead of relying on fixed rules

---

## ⚠ Disclaimer

This project is **purely a simulation and research prototype**.  
It is **not intended for real-world medical use** and should **not** be used for clinical decision-making.

---

## 🌱 Future Work

- Multi-drug interaction modeling
- Patient-specific parameter randomization
- Offline RL with clinical datasets
- Safety-constrained RL (hard toxicity limits)
- Integration with digital twin models

---

## ✨ Motivation

This project was inspired by the idea that **medicine doesn’t have to be static**.  
By combining **Reinforcement Learning** with **biological system modeling**, we can explore safer, more personalized treatment strategies — at least in simulation.

---

## 👤 Author

**Harsh Jain**  
B.Tech AIML  
Exploring Reinforcement Learning, Computational Neuroscience, and AI for Healthcare

---

⭐ If you find this project interesting, consider starring the repo!
