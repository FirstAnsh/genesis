<h1 align="center">Genesis: Specialized AI Agent</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white" alt="Weights & Biases" />
</div>

<br/>

> **Genesis** is an autonomous machine learning agent designed to process input commands, execute model predictions, and automatically track experiment metrics. 

## 📖 Overview

Genesis bridges the gap between raw machine learning models and automated execution. It is engineered to handle inputs (including audio commands), run inference using stored models, and dynamically track evaluation metrics like species classification and feature importance. 

**Key Features:**
*   **ML Inference Engine:** Executes automated predictions and processes data seamlessly (`predict.py`).
*   **Experiment Tracking:** Deep integration with Weights & Biases to log runs, track performance, and maintain state (`tracker.py`).
*   **Automated Evaluation:** Generates and stores visual metrics like confusion matrices and feature importance plots.

---

## 🏗️ Architecture & Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python 3.x | Backend logic and orchestration (`agent.py`) |
| **Tracking Provider** | Weights & Biases | Real-time experiment and metric logging |
| **Framework** | Custom Architecture | Agent loop and workflow management |
| **Environment** | python-dotenv | Secure API key configuration |

---

## 🚀 Quick Start

Follow these steps to run Genesis locally on your machine. 

### Prerequisites
*   Python 3.8+
*   Weights & Biases API Key (WANDB_API_KEY)
*   *(Optional)* Hugging Face / OpenAI API Key if using external LLMs

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/FirstAnsh/genesis.git](https://github.com/FirstAnsh/genesis.git)
   cd genesis
