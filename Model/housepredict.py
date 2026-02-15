import numpy as np;
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error;
import math;
import joblib;

#Loading the dataset
df = pd.read_csv('train.csv')

#Preprocessing the data
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'SalePrice']
df = df[features].dropna()

#train set
x = df.drop('SalePrice', axis=1)
#test set
y = df['SalePrice']


#split data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model =RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train,y_train)

#Predicting the test set results
y_pred = model.predict(x_test)
#Evaluating the model
meanSE = mean_squared_error(y_test, y_pred)


#saving the model
joblib.dump(model, 'house_price_predictor.pkl')


print("\n" + "="*50)
print("Model Evaluation Metrics:")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

rmse = math.sqrt(meanSE)
print(f"Root Mean Squared Error: {rmse:.2f}")