

---

# 📘 Gradient Boosting Regression – From Scratch

## 📌 Project Overview

This project presents a **from-scratch implementation of Gradient Boosting Regression** using Python. The goal is to understand how boosting works internally by building the algorithm without relying on high-level machine learning libraries such as `sklearn`'s built-in gradient boosting models.

The model is trained and evaluated on the **Boston Housing Dataset**, with careful attention to performance optimization, interpretability, and clean implementation.

---

## 🎯 Objectives

* Implement **Gradient Boosting Regression** from scratch
* Build a **custom Decision Tree Regressor**
* Apply **squared error loss**
* Improve performance using:

  * Learning rate
  * Subsampling (stochastic boosting)
  * Feature subsampling
  * Early stopping
* Evaluate using **RMSE** and **R² score**
* Visualize training loss

---

## 📂 Project Structure

```
📁 Gradient_Boosting_From_Scratch
│
├── Gradient_Boosting_From_Scratch.ipynb   # Main notebook
└── README.md                              # Project documentation
```

---

## ⚙️ Technologies Used

* **Python 3**
* **NumPy**
* **Matplotlib**
* **Scikit-learn** (only for dataset loading & evaluation metrics)

---

## 📊 Dataset

* **Boston Housing Dataset**
* Loaded manually (since `load_boston()` is deprecated in newer sklearn versions)
* Contains 506 samples with 13 numerical features
* Target variable: Median house price

---

## 🧠 Model Architecture

### 1. Decision Tree (Base Learner)

* Built from scratch
* Uses variance reduction (MSE) for splitting
* Supports:

  * Max depth
  * Minimum samples per split
  * Feature subsampling

### 2. Gradient Boosting Framework

* Sequential learning using residuals
* Learning rate (shrinkage)
* Early stopping to prevent overfitting
* Stochastic subsampling for stability

---

## 📈 Evaluation Metrics

The model is evaluated using:

### ✔ Root Mean Squared Error (RMSE)

Measures prediction error magnitude.

### ✔ R² Score (Accuracy for Regression)

Indicates how well the model explains variance in data.

---

## 📊 Typical Results

| Metric     | Value        |
| ---------- | ------------ |
| Train RMSE | ~1.7 – 2.0   |
| Test RMSE  | ~2.5 – 3.0   |
| Train R²   | ~0.93 – 0.95 |
| Test R²    | ~0.84 – 0.88 |

> These results indicate strong predictive performance with minimal overfitting.

---

## 📉 Training Curve

The notebook includes a visualization of:

* Training loss (MSE) vs number of estimators
* Helps analyze convergence and early stopping behavior

---

## ▶️ How to Run

### Step 1: Install dependencies

```bash
pip install numpy matplotlib scikit-learn
```

### Step 2: Open the notebook

```bash
jupyter notebook Gradient_Boosting_From_Scratch.ipynb
```

### Step 3: Run all cells

The notebook will:

* Load the dataset
* Train the model
* Display evaluation metrics
* Plot training loss

---

## 🧪 Key Learning Outcomes

* Deep understanding of gradient boosting internals
* Hands-on experience building ML models from scratch
* Improved understanding of bias–variance tradeoff
* Practical exposure to regression evaluation metrics

---

## ✅ Conclusion

This project demonstrates a **complete, optimized, and interpretable implementation of Gradient Boosting Regression** without relying on black-box libraries. It serves as a strong foundation for understanding advanced ensemble learning techniques used in real-world machine learning systems.

---

## 📬 Author

**Name:** *D.V.Guru Prakash*
**Course:** Machine Learning
**Institution:** *(SRM AP)*

---



