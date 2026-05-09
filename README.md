#  Diabetes Prediction using ANN + SHAP

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

##  Overview

This project builds an **Artificial Neural Network (ANN)** to predict diabetes using the **Pima Indians Diabetes Dataset**, and enhances interpretability using **SHAP (SHapley Additive Explanations)**.

The goal is not only to achieve good prediction accuracy but also to **understand *why* the model makes decisions**.

---

##  Key Features

- Data preprocessing and cleaning  
- ANN model for binary classification  
- Feature scaling and normalization  
- Model performance evaluation  
- Explainable AI using SHAP  
- Global + individual prediction explanations  

---

##  Project Structure
```text
Diabetes_ANN_Project/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── preprocessing/
│ ├── model/
│ ├── explainability/
│ 
│
├── outputs/
│ ├── models/
│ ├── plots/
│ └── results/
│
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```
---

##  Methodology

###  Data Preprocessing
- Replaced invalid zero values with median  
- Handled missing values  
- Normalized features using StandardScaler  
- Performed train-test split  

---

###  Model (ANN)
- Dense layers with ReLU activation  
- Dropout to prevent overfitting  
- Sigmoid activation for binary classification  

---

###  Explainable AI (SHAP)
- Global feature importance  
- Patient-level explanation  
- Interpretation of model decisions  

---

##  Results

- Final Test Accuracy: ~72–76%  
- Most important features:
  - Glucose  
  - BMI  
  - Age  

---

##  Outputs

### SHAP Summary Plot
![SHAP Summary](‪C:\CV\computer_vision_pima_diabetes\outputs\results\shap_patient_explanation.png)

---

### SHAP Patient Explanation
![SHAP Patient](C:\CV\computer_vision_pima_diabetes\outputs\results\shap_summary_plot.png)

---

##  How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
```
---

### 2. Run the project
```bash
python main.py
```

