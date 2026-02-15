# 🏠 House Price Prediction Model Documentation

## 📋 Overview

This machine learning model predicts house sale prices based on four key features using a Random Forest Regressor algorithm.

---

## 🎯 Model Purpose

**Goal:** Predict the sale price of houses based on physical characteristics

**Algorithm:** Random Forest Regressor

**Use Case:** Real estate price estimation for buyers, sellers, and real estate agents

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

## 🔧 Code Breakdown

### 1️⃣ **Import Libraries**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
```

**Purpose of each library:**
- **numpy:** Numerical operations and arrays
- **pandas:** Data manipulation and CSV reading
- **train_test_split:** Split data into training and testing sets
- **RandomForestRegressor:** The ML algorithm
- **mean_squared_error:** Evaluate model accuracy
- **joblib:** Save the trained model to a file

---

### 2️⃣ **Load Dataset**

```python
df = pd.read_csv('train.csv')
```

**What it does:** Loads the CSV file into a pandas DataFrame

**Result:** A table with ~1,460 rows (houses) and 81 columns (features)

---

### 3️⃣ **Preprocess Data**

```python
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'SalePrice']
df = df[features].dropna()
```

**What it does:**
1. **Select only relevant columns** from the 81 available features
2. **Remove missing values** using `dropna()`

**Why:** 
- Simplifies the model (only 4 input features instead of 80)
- Ensures no missing data that could cause errors
- Focuses on the most important predictors of house price

---

### 4️⃣ **Split Features and Target**

```python
# Input features (X)
x = df.drop('SalePrice', axis=1)

# Target variable (y)
y = df['SalePrice']
```

**Result:**

**X (Input):**
```
   OverallQual  GrLivArea  GarageCars  TotalBsmtSF
0            7       1710           2          856
1            6       1262           2          1262
2            7       1786           2          920
...
```

**y (Output):**
```
0    208500
1    181500
2    223500
...
```

---

### 5️⃣ **Split Data into Training & Testing Sets**

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

**Parameters explained:**
- **`test_size=0.2`:** 20% of data for testing, 80% for training
- **`random_state=42`:** Ensures the split is reproducible (same random split every time)

**Result:**
- **Training set:** ~1,168 houses (80%) - Used to teach the model
- **Testing set:** ~292 houses (20%) - Used to evaluate the model

**Why split?**
- Train the model on some data
- Test it on "unseen" data to check if it generalizes well
- Prevents overfitting

---

### 6️⃣ **Create and Train the Model**

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
```

**What is Random Forest Regressor?**
- An ensemble learning method
- Creates 100 decision trees (like voting by committee)
- Each tree makes a prediction, then averages them
- More robust than a single decision tree

**Parameters:**
- **`n_estimators=100`:** Creates 100 decision trees
- **`random_state=42`:** Ensures reproducible results

**What `.fit()` does:**
- Learns patterns from the training data
- Finds relationships between features and prices
- Example: "Houses with higher OverallQual tend to cost more"

---

### 7️⃣ **Make Predictions**

```python
y_pred = model.predict(x_test)
```

**What it does:** Uses the trained model to predict prices for the test set

**Example:**
```python
# Test data (actual house):
# OverallQual=7, GrLivArea=2000, GarageCars=2, TotalBsmtSF=1200

# Model predicts: $234,567
# Actual price: $238,000
# Error: $3,433
```

---

### 8️⃣ **Evaluate Model Performance**

```python
meanSE = mean_squared_error(y_test, y_pred)
```

**What is Mean Squared Error (MSE)?**
- Measures average squared difference between predictions and actual values
- Lower MSE = Better model
- Formula: `MSE = average((actual - predicted)²)`

**Example calculation:**
```python
Actual:    [200000, 150000, 300000]
Predicted: [210000, 145000, 295000]
Errors:    [10000,  -5000,   5000]
Squared:   [100M,   25M,     25M]
MSE:       50M (average of squared errors)
RMSE:      7,071 (√MSE) - average error in dollars
```

**Typical values:**
- MSE: ~1-2 billion
- RMSE: ~$30,000-$45,000 (typical prediction error)

---

### 9️⃣ **Save the Model**

```python
joblib.dump(model, 'house_price_predictor.pkl')
```

**What it does:** Saves the trained model to a file

**Why?**
- Don't need to retrain every time
- Can load it in Flask API for predictions
- The `.pkl` file contains all the learned patterns

**File size:** ~10-50 MB (depends on model complexity)

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

## 📈 Model Performance Metrics

### What to Add After Training:

After running the model, add these metrics to your documentation:

```python
from sklearn.metrics import mean_squared_error, r2_score
import math

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (more interpretable)
rmse = math.sqrt(mse)

# R² Score (0-1, higher is better)
r2 = r2_score(y_test, y_pred)

print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
```

**Example output:**
```
MSE: $1,234,567,890
RMSE: $35,136
R² Score: 0.8234
```

**What these mean:**
- **RMSE:** On average, predictions are off by ~$35,000
- **R² Score:** Model explains 82.34% of price variance (good!)

---

## 🎯 Feature Importance

**Which features matter most?**

```python
# Get feature importance
importances = model.feature_importances_
features_names = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

for name, importance in zip(features_names, importances):
    print(f"{name}: {importance:.4f}")
```

**Typical results:**
```
OverallQual: 0.4521  (45.21% importance)
GrLivArea:   0.3234  (32.34% importance)
GarageCars:  0.1123  (11.23% importance)
TotalBsmtSF: 0.1122  (11.22% importance)
```

**Interpretation:** Overall quality is the strongest predictor of price

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
2. **No feature scaling:** May benefit from normalization
3. **Simple preprocessing:** Doesn't handle outliers or feature engineering
4. **No hyperparameter tuning:** Using default Random Forest settings

### Potential Improvements:
- Add more relevant features (YearBuilt, Neighborhood, LotArea, etc.)
- Feature engineering (e.g., Age = CurrentYear - YearBuilt)
- Grid search for optimal hyperparameters
- Try other algorithms (XGBoost, Gradient Boosting)
- Handle outliers and feature scaling

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

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial model with 4 features |

---

## 👤 Author & Usage

**Created by:** [Kalana Malhara]
**Purpose:** Educational project for learning ML and web development

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


