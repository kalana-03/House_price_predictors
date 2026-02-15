# 🏠 House Price Predictor

## 📋 Overview

This machine learning model predicts house sale prices based on four key features using a Random Forest Regressor algorithm.

---

## 🎯 Model Purpose

**Goal:** Predict the sale price of houses based on physical characteristics

**Algorithm:** Random Forest Regressor

---

## 📊 Dataset

**Source:** `train.csv` (Kaggle House Prices dataset)

**Features Used:**

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `OverallQual` | Overall material and finish quality | Integer | 1-10 |
| `GrLivArea` | Above grade (ground) living area | Integer | Square feet |
| `GarageCars` | Size of garage in car capacity | Integer | 0-4 |
| `TotalBsmtSF` | Total square feet of basement area | Integer | Square feet |
| `SalePrice` | Sale price of the house (TARGET) | Integer | USD ($) |

---


## 🔄 Complete Workflow

```
📁 Load CSV
    ↓
🧹 Clean Data (select features, remove missing values)
    ↓
✂️ Split Data (80% train, 20% test)
    ↓
🌳 Train Model (Random Forest with 100 trees)
    ↓
🎯 Make Predictions on test set
    ↓
📊 Evaluate Performance (MSE/RMSE)
    ↓
💾 Save Model to .pkl file
    ↓
🚀 Use in Flask API for real-time predictions
```

---
## 🛠️ How to Use the Model

### In Python:
```python
import joblib
import numpy as np

# Load the saved model
model = joblib.load('house_price_predictor.pkl')

# Make a prediction
input_data = np.array([[7, 2000, 2, 1200]])  # [Quality, Area, Cars, Basement]
prediction = model.predict(input_data)[0]

print(f"Predicted Price: ${prediction:,.2f}")
# Output: Predicted Price: $234,567.89
```

### In Flask API:
See the Flask API documentation (`app.py`)

### In React Frontend:
See the React component documentation (`HousePricePrediction.tsx`)

---

## 🔍 Model Limitations

### Current Limitations:
1. **Limited features:** Only uses 4 out of 81 available features
3. **Simple preprocessing:** Doesn't handle outliers or feature engineering
4. **No hyperparameter tuning:** Using default Random Forest settings


---

## 📚 Dependencies

```bash
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
```

Install all:
```bash
pip install numpy pandas scikit-learn joblib
```

---

## 🎓 Key Machine Learning Concepts

### Random Forest
- **Type:** Ensemble learning method
- **How it works:** Builds multiple decision trees and averages their predictions
- **Pros:** Handles non-linear relationships, resistant to overfitting, no feature scaling needed
- **Cons:** Less interpretable than linear models, slower than simple algorithms

### Train-Test Split
- **Purpose:** Evaluate model on unseen data
- **Ratio:** 80/20 is common (can also use 70/30 or 90/10)
- **Why:** Prevents overfitting and gives realistic performance estimate

### Mean Squared Error
- **Purpose:** Measures prediction accuracy
- **Calculation:** Average of squared differences
- **Why squared?** Penalizes large errors more than small ones

---


## 👤 Purpose 
Educational project for learning ML and web development

---

## 🔗 Related Files

- **Training Script:** `housepredictor.py`
- **Model File:** `house_price_predictor.pkl`
- **Flask API:** `app.py`
- **React Frontend:** `HousePricePrediction.tsx`
- **Dataset:** `train.csv`

---

## 🚀 Quick Start

1. **Train the model:**
   ```bash
   python housepredictor.py
   ```

2. **Start Flask API:**
   ```bash
   cd Model
   python app.py
   ```

3. **Start React app:**
   ```bash
   cd frontend
   npm start
   ```

4. **Make predictions:**
   - Open browser to `http://localhost:3000`
   - Enter house features
   - Get instant price prediction!

---


