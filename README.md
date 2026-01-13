# Veracity NLP: Fake News Detection System

> **A High-Precision Machine Learning System for Misinformation Analysis.**
> *Powered by Linear SVM & TF-IDF with 98.8% Accuracy.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![SVM](https://img.shields.io/badge/Algorithm-Linear%20SVM-green?style=for-the-badge)

##  Project Overview
**Veracity NLP** is an end-to-end machine learning pipeline designed to classify news articles as **Real** or **Fake** with high linguistic precision. Unlike basic classifiers, this system focuses on **stylometric features** (writing style) rather than memorizing sources, ensuring unbiased analysis.

The project features a full **Streamlit Dashboard** for real-time inference and a robust training pipeline that benchmarks multiple algorithms based on academic standards.

---

## 🔬 Model Architecture & Methodology
This project implements a robust machine learning pipeline utilizing a **Model Factory Pattern**. The selection of algorithms was driven by a comparative analysis of high-performance models recommended in recent academic literature.

### 1. Baseline Model: Logistic Regression
* **Role:** Serves as the primary statistical baseline for binary classification.
* **Scientific Rationale:** Selected for its efficiency in handling linearly separable textual features and establishing a performance benchmark (Reference: Studies S3 & S5).

### 2. Maximum Margin Classifier: Support Vector Machine (SVM) 🏆
* **Role:** The core engine of this project. It acts as a powerful discriminative classifier that finds the optimal hyperplane to separate real vs. fake news vectors.
* **Scientific Rationale:** SVM is a cornerstone in fake news research due to its superior performance in high-dimensional spaces created by text vectorization (Reference: Studies S1, S3, & S5).

### 3. Ensemble Learning Frameworks
We incorporated tree-based ensemble methods to capture non-linear relationships:
* **Random Forest:** Utilized to reduce overfitting through bagging and multiple decision tree estimators.
* **Gradient Boosting (XGBoost):** Following the research of *Alshuwaier & Alsulaiman (2024)*, Gradient Boosting frameworks were selected for their proven ability to achieve high accuracy in complex classification tasks.

### 4. Feature Engineering Strategy
* **Technique:** TF-IDF (Term Frequency-Inverse Document Frequency) combined with **N-grams (Bigrams)**.
* **Effectiveness:** This combination transforms raw text into numerical vectors while preserving local context (word pairs), cited as a highly effective approach for estimating news credibility.

---

## Scientific References
The methodology and algorithm selection for this project are informed by the following peer-reviewed sources:

> 1. **Alshuwaier, F. A., & Alsulaiman, F. A. (2024).** *"Fake News Detection Using Machine Learning and Deep Learning Algorithms: A Comprehensive Review and Future Perspectives."* Computers, 14(9), 394.
> 2. **ScienceDirect Research (2025).** *"Machine Learning and NLP Advancements for Information Credibility."*

---

## Benchmarking Results
After rigorous testing, **Linear SVM** emerged as the champion model, balancing speed and precision:

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **SVM (Linear)** | **98.8%** | **0.988** | **~2 sec** |
| Logistic Regression | 98.3% | 0.983 | ~2 sec |
| XGBoost | 98.2% | 0.982 | ~540 sec |
| Random Forest | 97.5% | 0.975 | ~33 sec |

---

##  Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Veracity-NLP-Fake-News-Detector.git](https://github.com/YOUR_USERNAME/Veracity-NLP-Fake-News-Detector.git)
   cd Veracity-NLP-Fake-News-Detector

   2.Install dependencies:
pip install -r requirements.txt

3.Run the Dashboard:
streamlit run app.py

Author:

Yahya Abu Zahra - Computer Engineering Undergraduate (AI Track)
