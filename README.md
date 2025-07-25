# Predicting-Heart-Disease-Risk-Using-Supervised-Machine-Learning-Algorithms

# ❤️ Heart Disease Prediction using Machine Learning

## 👩‍💻 Author

**Maria-Daniela Munteanu**  
*Faculty of Automatic Control and Computer Engineering*  
*Technical University "Gheorghe Asachi" of Iași*  

## 🧠 Project Overview

This project explores the application of **supervised machine learning** techniques to predict the presence of heart disease based on real clinical attributes. Developed as an academic research project, the study investigates two popular classification models — **K-Nearest Neighbors (K-NN)** and **Random Forest** — and evaluates their performance using a real-world medical dataset.

---

## 🩺 Problem Statement

Cardiovascular diseases (CVDs) are the **leading global cause of death**, accounting for approximately 17.9 million deaths annually (31% of global deaths). Early diagnosis is critical for effective treatment and reducing mortality.

The main goal of this project is to **predict the likelihood of heart disease** using clinical variables, leveraging machine learning algorithms to assist in early detection and decision support.

---

## 🗂️ Dataset Description

- **Dataset Source**: Kaggle, Heart Failure Prediction dataset  
  [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Samples**: 918 unique patients
- **Target Variable**: `HeartDisease` (1 = present, 0 = absent)
- **Features** (11 clinical attributes):
  - Age
  - Sex
  - ChestPainType
  - RestingBP
  - Cholesterol
  - FastingBS
  - RestingECG
  - MaxHR
  - ExerciseAngina
  - Oldpeak
  - ST_Slope

ℹ️ The data was cleaned from an initial 1190 entries by removing 272 duplicates:contentReference[oaicite:0]{index=0}.

---

## 📊 Exploratory Data Analysis (EDA)

- A **class imbalance** was observed: 508 positive vs. 410 negative cases.
- A **correlation matrix** was computed to assess multicollinearity. No feature pairs exceeded a correlation threshold of 0.9, confirming no need for feature.
- Histograms and distribution plots were used to understand the characteristics of each feature.

---

## 🧪 Machine Learning Models

### 🔹 K-Nearest Neighbors (K-NN)
- A non-parametric, instance-based model
- Tested with **k=5** and **k=10**
- Distance metric: Euclidean

### 🔹 Random Forest
- Ensemble of decision trees with bootstrap sampling
- Tested with **50** and **200** estimators
- Feature selection based on **Gini Impurity**
- Handles missing data and is robust to noise

---

## 📈 Model Evaluation

Performance was measured using:
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1 Score**

### 🧮 Results Summary:

| Model                  | Precision | Recall   | F1 Score | Accuracy |
|------------------------|-----------|----------|----------|----------|
| K-NN (k=5)             | 0.83654   | 0.89691  | 0.86567  | 0.85246  |
| K-NN (k=10)            | 0.88542   | 0.87629  | 0.88083  | 0.87432  |
| Random Forest (50)     | 0.84615   | 0.90722  | 0.87562  | 0.86339  |
| Random Forest (200)    | 0.85294   | 0.89691  | 0.87437  | 0.86339  |

📌 The best performance was achieved by **K-NN (k=10)** in terms of accuracy and **Random Forest** in terms of recall.

---

## 🔍 Insights & Observations

- K-NN provided quick, interpretable predictions and performed well with proper parameter tuning.
- Random Forest offered better robustness, generalization, and sensitivity to positive cases.
- Feature distributions and correlations suggested the dataset was well-suited for both models without heavy feature engineering.

---

## 🚀 Potential Improvements

- 🔧 **Hyperparameter tuning** via grid search and cross-validation
- 🧼 **Advanced preprocessing**: scaling, encoding, and imputation
- 🧠 **Model stacking/ensembling** for performance boosts
- 📈 **Deploying** a web-based app for clinical decision support

---

## 📚 References

- [Springer Journal – ML for Heart Disease](https://link.springer.com/article/10.1007/s42979-023-02529-y)
- [MDPI Algorithms – Predictive Techniques](https://www.mdpi.com/1999-4893/17/2/78)
- [Heart Failure Registry Romania](https://revista.cardioportal.ro/arhiva/characteristics-of-patients-with-heart-failure-from-romania-enrolled-in-esc-hf-long-term-esc-hf-lt-registry/)
- [Kaggle – Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- [INO-MED Clinical Guide](https://ino-med.ro/docs/document-de-pozitie-insuficienta-cardiaca.pdf)

---

> This project was developed as part of the “Embedded Control Systems” course and showcases the synergy between data science and healthcare through interpretable machine learning solutions.
