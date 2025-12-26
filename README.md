
# Gradient Boosting Regression – From Scratch

## 📌 Project Overview

This project implements **Gradient Boosting Regression from scratch** using Python.  
The main objective is to understand how gradient boosting works internally by building the algorithm manually without using high-level machine learning libraries such as `sklearn`’s built-in models.

The model is trained and evaluated using the **Boston Housing Dataset**, a classic regression dataset commonly used for educational and benchmarking purposes.

---

## 🎯 Objectives

- Implement Gradient Boosting from scratch  
- Build a custom Decision Tree Regressor  
- Use Mean Squared Error (MSE) as the loss function  
- Improve performance using:
  - Learning rate (shrinkage)
  - Subsampling (stochastic boosting)
  - Feature subsampling
  - Early stopping  
- Perform basic Exploratory Data Analysis (EDA)  
- Evaluate performance using **RMSE** and **R² Score**

---

## 📂 Project Structure

```

Gradient_Boosting_From_Scratch/
│
├── Gradient_Boosting_From_Scratch.ipynb
└── README.md

````

---

## ⚙️ Technologies Used

- Python 3  
- NumPy  
- Matplotlib  
- Scikit-learn (used only for dataset loading and evaluation metrics)

---

## 📊 Dataset Description

### Boston Housing Dataset

The Boston Housing dataset contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts.

**Dataset characteristics:**
- 506 total samples  
- 13 numerical input features  
- Target variable: Median value of owner-occupied homes  

### Features include:
- CRIM – Per capita crime rate  
- ZN – Proportion of residential land zoned  
- INDUS – Proportion of non-retail business acres  
- NOX – Nitric oxide concentration  
- RM – Average number of rooms  
- AGE – Proportion of owner-occupied units built before 1940  
- DIS – Distance to employment centers  
- RAD – Accessibility to radial highways  
- TAX – Property tax rate  
- PTRATIO – Pupil–teacher ratio  
- B – Proportion of Black population  
- LSTAT – Percentage of lower status population  

> ⚠️ Note: Although deprecated in recent sklearn versions, the Boston dataset is used here **strictly for educational purposes**.

---

## 🔍 Exploratory Data Analysis (EDA)

The following analysis is performed before training:

- Dataset shape and feature inspection  
- Summary statistics (mean, min, max, std)  
- Detection of feature distributions and variance  
- Correlation analysis between features and target variable  

EDA helps understand feature influence and improves model interpretability.

---

## 🧠 Model Architecture

### 1. Decision Tree Regressor (From Scratch)
- Uses Mean Squared Error (MSE) to determine best splits  
- Supports:
  - Maximum depth  
  - Minimum samples per split  
  - Feature subsampling  

### 2. Gradient Boosting Framework
- Sequentially trains trees on residuals  
- Applies learning rate to control update strength  
- Uses stochastic subsampling to improve generalization  
- Includes early stopping to avoid overfitting  

---

## 📈 Evaluation Metrics

### ✔ Root Mean Squared Error (RMSE)
Measures the average prediction error magnitude.

### ✔ R² Score (Regression Accuracy)
Indicates how well the model explains the variance in the target variable.

---

## 📊 Typical Results

| Metric | Value |
|------|------|
| Train RMSE | ~1.7 – 2.0 |
| Test RMSE | ~2.5 – 3.0 |
| Train R² | ~0.93 – 0.95 |
| Test R² | ~0.84 – 0.88 |

These results show strong predictive performance with minimal overfitting.

---

## 📉 Visualizations

The notebook includes:
- Training loss curve (MSE vs iterations)  
- Actual vs predicted price comparison  
- Feature distribution plots  

---

## ▶️ How to Run

### Step 1: Install required libraries
```bash
pip install numpy matplotlib scikit-learn
````

### Step 2: Run the notebook

```bash
jupyter notebook Gradient_Boosting_From_Scratch.ipynb
```

### Step 3: Execute all cells

The notebook will:

* Load and preprocess the dataset
* Perform exploratory data analysis
* Train the gradient boosting model
* Evaluate and visualize results

---

## 🧪 Learning Outcomes

* Strong understanding of gradient boosting fundamentals
* Hands-on experience implementing ML algorithms from scratch
* Improved understanding of regression evaluation metrics
* Ability to analyze and interpret model performance

---

## ✅ Conclusion

This project demonstrates a complete and optimized implementation of **Gradient Boosting Regression from scratch** using the Boston Housing dataset.
It highlights both theoretical understanding and practical implementation of ensemble learning methods.

---

## 👤 Author

**Name:** *D.V.Guru Prakash*
**Course:** Machine Learning
**Institution:** *SRM AP*

---

## ⭐ Optional Enhancements

* Feature importance visualization
* Cross-validation
* Hyperparameter tuning
* Comparison with sklearn’s GradientBoostingRegressor

---

