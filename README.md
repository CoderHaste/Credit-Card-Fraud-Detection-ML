# **💳  Credit Card Fraud Detection using Machine Learning**

### 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Credit card fraud datasets are highly imbalanced, making accurate detection challenging.

To address this, this project implements Random Forest and applies SMOTE (Synthetic Minority Oversampling Technique) to improve fraud detection performance and increase recall for fraudulent transactions.

The model performance is evaluated and compared with and without SMOTE to understand the impact of handling class imbalance.

### 🚀 Key Features

Data preprocessing and exploration<br/>
Handling imbalanced dataset using SMOTE<br/>
Random Forest model training<br/>
Performance comparison: Before vs After SMOTE<br/>
Evaluation using Precision, Recall, F1-score, and ROC-AUC<br/>
Focus on fraud detection recall (important in real-world banking systems)<br/>

### 🛠️ Tech Stack

Programming: Python<br/>
Libraries: Pandas, Scikit-learn, Matplotlib<br/>
Machine Learning: Random Forest<br/>
Imbalance Handling: SMOTE (Imbalanced-learn)<br/>

### 📊 Model Evaluation

The model is evaluated using industry-relevant metrics:<br/>
ROC-AUC Score: ~0.90+<br/>
High recall for fraud transactions<br/>
Improved detection after applying SMOTE<br/>
Balanced performance across precision and recall<br/>
This demonstrates the importance of handling imbalanced datasets in financial fraud detection systems.<br/>

### 📂 Project Structure
Credit-Card-Fraud-Detection-ML<br/>
│── creditcard.csv<br/>
│── fraud_detection.py<br/>
│── requirements.txt<br/>
│── README.md<br/>
▶️ How to Run the Project<br/>
1️⃣ Install dependencies<br/>
pip install -r requirements.txt<br/>
2️⃣ Run the model<br/>
python fraud_detection.py<br/>
### 🧠 Learning Outcomes

Understanding imbalanced datasets in real-world scenarios<br/>
Using SMOTE for improving minority class prediction<br/>
Evaluating ML models beyond accuracy<br/>
Building production-style ML project structure

### 🔮 Future Improvements

Hyperparameter tuning for optimization<br/>
Testing with XGBoost & Gradient Boosting<br/>
Model deployment using Streamlit<br/>
Real-time fraud detection pipeline

### 👨‍💻 Author

Prateek Manjunath<br/>
Data Science & Machine Learning Enthusiast
