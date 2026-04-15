# 💳 AI Fraud Detection System

## Problem Statement
Credit card fraud is a major issue in the financial industry, causing significant financial losses and security risks. This project aims to build a machine learning system that can accurately detect fraudulent transactions based on historical data.

The application allows users to test transactions and instantly determine whether they are fraudulent or legitimate, helping in early detection and prevention.

## Tech Stack
- **Python 3.9**
- **Pandas** — data loading and preprocessing
- **Scikit-learn** — Random Forest model
- **Streamlit** — interactive web UI
- **Seaborn & Matplotlib** — data visualization

## Features
- 📊 Balanced dataset using upsampling
- 🎯 Fraud detection using Random Forest classifier
- 🔍 Test transactions using real dataset samples
- 📈 Prediction probability display
- 📉 Confusion matrix and classification report
- 📊 Dataset class distribution visualization

## Screenshots
![Main Dashboard](screenshots/Screenshot%20-main%20dashboard.png)
![Fraud Detected](screenshots/Screenshot%20-fraud%20dectected.png)
![Legit Transaction](screenshots/Screenshot%20-legit%20transaction.png)

## How to Run

1. Clone the repo:
git clone https://github.com/YOUR_USERNAME/ai-fraud-detection-system.git
cd ai-fraud-detection-system

2. Install dependencies:
pip install streamlit pandas scikit-learn seaborn matplotlib

3. Run the app:
python3 -m streamlit run fraud_app.py
