# Smart Irrigation Optimization System using Reinforcement Learning and MLOps

## Overview

The Smart Irrigation Optimization System is an AI-powered irrigation decision system that uses Reinforcement Learning (RL) to intelligently decide when irrigation should be performed based on environmental conditions.

The project simulates agricultural field conditions such as:

* Soil Moisture
* Temperature
* Rainfall

Using these environmental states, RL agents learn optimal irrigation policies that:

* Reduce unnecessary water usage
* Improve irrigation efficiency
* Maximize agricultural sustainability
* Support smart farming automation

The project also integrates complete MLOps practices including:

* MLflow Experiment Tracking
* FastAPI Deployment
* Docker Containerization
* GitHub Actions CI/CD
* Logging and Monitoring

---

# SDG Mapping

This project addresses:

## SDG 6 — Clean Water and Sanitation

The system optimizes irrigation decisions to minimize water wastage and promote efficient water resource utilization.

## SDG 9 — Industry, Innovation and Infrastructure

The project demonstrates intelligent automation using Reinforcement Learning and scalable MLOps infrastructure.

## SDG 13 — Climate Action

By reducing unnecessary irrigation, the project contributes toward sustainable agricultural practices and environmental conservation.

---

# Problem Statement

Traditional irrigation systems often:

* Overuse water
* Ignore changing environmental conditions
* Operate on fixed schedules
* Waste resources during rainfall
* Require manual intervention

The goal of this project is to build an intelligent irrigation agent capable of learning optimal watering strategies automatically using Reinforcement Learning.

---

# Objectives

The major objectives of this project are:

* Design a Reinforcement Learning environment for irrigation optimization
* Implement multiple RL algorithms
* Compare algorithm performance
* Perform hyperparameter tuning
* Deploy the model using FastAPI
* Track experiments using MLflow
* Containerize the application using Docker
* Automate testing using GitHub Actions CI/CD

---

# Technologies Used

| Technology             | Purpose                     |
| ---------------------- | --------------------------- |
| Python                 | Core Programming Language   |
| Reinforcement Learning | Intelligent Decision Making |
| Q-Learning             | RL Algorithm                |
| SARSA                  | RL Algorithm                |
| DQN                    | Deep Reinforcement Learning |
| PyTorch                | Neural Network Backend      |
| FastAPI                | REST API Deployment         |
| MLflow                 | Experiment Tracking         |
| Docker                 | Containerization            |
| GitHub Actions         | CI/CD Automation            |
| Matplotlib             | Visualization               |
| Pandas                 | Data Analysis               |

---

# Project Architecture

```text
User Input
    ↓
FastAPI REST API
    ↓
RL Decision Engine
    ↓
Q-Learning / SARSA / DQN
    ↓
Irrigation Prediction
    ↓
Prediction Logging
    ↓
MLflow Tracking
    ↓
Docker Container
    ↓
GitHub CI/CD Pipeline
```

---

# Reinforcement Learning Environment

The environment simulates agricultural field conditions.

## State Space

Each state consists of:

| Feature       | Description         |
| ------------- | ------------------- |
| Soil Moisture | Dry / Medium / Wet  |
| Temperature   | Low / Medium / High |
| Rainfall      | No Rain / Rain      |

Example state:

```python
(0, 1, 0)
```

Where:

* 0 → Dry Soil
* 1 → Medium Temperature
* 0 → No Rainfall

---

## Action Space

| Action | Meaning         |
| ------ | --------------- |
| 0      | Do Not Irrigate |
| 1      | Irrigate        |

---

## Reward System

The reward function encourages:

* Water conservation
* Smart irrigation decisions
* Avoidance of overwatering
* Efficient irrigation timing

Example:

| Situation                  | Reward   |
| -------------------------- | -------- |
| Irrigate dry soil          | Positive |
| Irrigate during rainfall   | Negative |
| Avoid unnecessary watering | Positive |

---

# Algorithms Implemented

## 1. Q-Learning

Q-Learning is an off-policy Reinforcement Learning algorithm that learns the optimal action-value function.

### Features

* Simple implementation
* Fast convergence
* Strong performance in discrete environments

### Performance

Q-Learning achieved strong average rewards and efficient water usage.

---

## 2. SARSA

SARSA is an on-policy Reinforcement Learning algorithm.

### Features

* Learns based on current policy
* More stable learning behavior
* Safer exploration strategy

### Performance

SARSA achieved highly stable performance with competitive rewards.

---

## 3. Deep Q Network (DQN)

DQN combines Q-Learning with neural networks using PyTorch.

### Features

* Deep Reinforcement Learning
* Neural Network approximation
* Handles larger state spaces

### Performance

DQN successfully learned irrigation policies but required more tuning and training time.

---

# Hyperparameter Tuning

Hyperparameter tuning was performed to improve agent performance.

## Parameters Tuned

| Parameter               | Description              |
| ----------------------- | ------------------------ |
| Learning Rate (Alpha)   | Learning speed           |
| Discount Factor (Gamma) | Future reward importance |
| Epsilon                 | Exploration rate         |

## Sample Tuning Results

| Experiment | Alpha | Gamma | Epsilon | Average Reward |
| ---------- | ----- | ----- | ------- | -------------- |
| 1          | 0.10  | 0.95  | 0.20    | 66.49          |
| 2          | 0.05  | 0.99  | 0.10    | 71.63          |
| 3          | 0.20  | 0.90  | 0.30    | 58.21          |
| 4          | 0.15  | 0.95  | 0.05    | 60.51          |

Best configuration:

```text
Alpha = 0.05
Gamma = 0.99
Epsilon = 0.10
```

---

# Model Evaluation

## Final Algorithm Comparison

| Algorithm  | Average Reward |
| ---------- | -------------- |
| Q-Learning | 70.34          |
| SARSA      | 71.58          |
| DQN        | 45.21          |

## Observation

* SARSA achieved the highest stability.
* Q-Learning achieved strong overall performance.
* DQN required additional tuning due to neural network complexity.

---

# MLflow Integration

MLflow was integrated for experiment tracking.

## Features Logged

* Hyperparameters
* Episode rewards
* Average rewards
* Maximum rewards
* Reward graphs
* Model artifacts

## Experiments Created

* Q_Learning_MLflow
* SARSA_MLflow
* DQN_MLflow

---

# FastAPI Deployment

A REST API was developed using FastAPI.

## API Endpoint

```http
POST /predict
```

## Example Input

```json
{
  "soil_moisture": 0,
  "temperature": 1,
  "rainfall": 0
}
```

## Example Output

```json
{
  "state": [0,1,0],
  "action": 1,
  "decision": "Irrigate"
}
```

---

# Prediction Logging

All API predictions are logged into:

```text
logs/predictions.log
```

Example log:

```text
State: (0, 0, 0) | Action: 1 | Decision: Irrigate
```

---

# Docker Integration

The entire application was containerized using Docker.

## Features

* Platform-independent deployment
* Simplified environment setup
* Portable execution
* Scalable deployment support

## Docker Commands

### Build and Run

```bash
docker compose up --build
```

### Stop Containers

```bash
docker compose down
```

---

# GitHub Actions CI/CD

Continuous Integration (CI) was implemented using GitHub Actions.

## Pipeline Features

* Automatic dependency installation
* Automated environment testing
* Automated agent testing
* Trigger on every push to main branch

## Workflow File

```text
.github/workflows/python-app.yml
```

---

# Project Structure

```text
Smart_Irrigation/
│
├── api/
├── experiments/
├── logs/
├── models/
├── plots/
├── sim/
├── tests/
├── screenshots/
├── train_ql.py
├── train_sarsa.py
├── train_dqn.py
├── train_ql_mlflow.py
├── train_sarsa_mlflow.py
├── train_dqn_mlflow.py
├── compare_all.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Setup Instructions

## Step 1 — Clone Repository

```bash
git clone <repository_link>
```

---

## Step 2 — Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 3 — Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Train Q-Learning

```bash
python train_ql.py
```

## Train SARSA

```bash
python train_sarsa.py
```

## Train DQN

```bash
python train_dqn.py
```

---

# Run MLflow

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# Run FastAPI

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# Run Docker

```bash
docker compose up --build
```

---

# Screenshots

## MLflow Dashboard

(Add Screenshot)

## FastAPI Swagger UI

(Add Screenshot)

## Docker Running

(Add Screenshot)

## GitHub Actions CI/CD

(Add Screenshot)

## Reward Curves

(Add Screenshot)

---

# Challenges Faced

* Designing the RL reward function
* Handling environment state transitions
* Managing Docker dependency issues
* Fixing CI/CD compatibility problems
* Integrating MLflow with RL workflows
* Stabilizing DQN performance

---

# Learning Outcomes

Through this project, the following concepts were learned:

* Reinforcement Learning fundamentals
* Q-Learning and SARSA implementation
* Deep Reinforcement Learning using DQN
* Hyperparameter tuning
* MLflow experiment tracking
* FastAPI deployment
* Docker containerization
* CI/CD automation using GitHub Actions
* MLOps workflow integration

---

# Future Improvements

Possible future enhancements:

* Integration with real IoT sensors
* Real weather API integration
* Continuous online learning
* Kubernetes deployment
* Cloud deployment on AWS/GCP
* Advanced Deep RL algorithms
* Real agricultural dataset integration

---

# Conclusion

The Smart Irrigation Optimization System successfully demonstrates the integration of Reinforcement Learning and MLOps to solve a real-world sustainability problem.

The project achieved:

* Intelligent irrigation optimization
* Efficient water management
* Multi-model RL comparison
* End-to-end MLOps deployment pipeline
* Automated experiment tracking and CI/CD

This project showcases both strong AI implementation skills and modern MLOps engineering practices.

---

# Authors

* Goutham T G
* Team Members

---

# License

This project is developed for academic and educational purposes.
