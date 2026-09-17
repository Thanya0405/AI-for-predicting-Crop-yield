# ============================================
# AI Model for Predicting Crop Yield
# Execution Phase Code
# ============================================

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 2. Sample / Synthetic Dataset ---
# (In real use, replace this section with actual CSV data)
np.random.seed(42)
n = 300

data = pd.DataFrame({
    'rainfall': np.random.uniform(200, 1000, n),
    'temperature': np.random.uniform(18, 35, n),
    'soil_nitrogen': np.random.uniform(0.5, 3.0, n),
    'ndvi': np.random.uniform(0.3, 0.9, n),
    'year': np.random.randint(2015, 2024, n)
})

# Simulated crop yield (target)
data['yield'] = (
    0.005 * data['rainfall']
    - 0.02 * data['temperature']**2
    + 10 * data['ndvi']
    + 2 * data['soil_nitrogen']
    + np.random.normal(0, 5, n)
)

print("Dataset preview:")
print(data.head())

# --- 3. Train-Test Split ---
X = data[['rainfall', 'temperature', 'soil_nitrogen', 'ndvi']]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 4. Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

# --- 5. Train and Evaluate ---
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results.append([name, rmse, mae, r2, mape])

# Results Table
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'R²', 'MAPE'])
print("\nModel Performance Summary:")
print(results_df)

# --- 6. Visualize Predicted vs Actual ---
best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred_best, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Line')
plt.xlabel("Actual Yield")
plt.ylabe
