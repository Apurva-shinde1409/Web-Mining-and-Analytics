# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Aggregated_Crimes_2020_2024.csv")

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.dropna(subset=['Primary Type'], inplace=True)

# Feature Engineering
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year
data['DayOfWeek'] = data.index.dayofweek  # Monday=0, Sunday=6
data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

# Encode Categorical Variables
data['Primary Type'] = data['Primary Type'].astype('category').cat.codes
data['Location Description'] = data['Location Description'].astype('category').cat.codes

# Aggregate Daily Crime Counts
daily_crime_counts = data.resample('D').size().rename('Crime_Count').to_frame()
daily_crime_counts['Day'] = daily_crime_counts.index.day
daily_crime_counts['Month'] = daily_crime_counts.index.month
daily_crime_counts['Year'] = daily_crime_counts.index.year
daily_crime_counts['DayOfWeek'] = daily_crime_counts.index.dayofweek
daily_crime_counts['IsWeekend'] = (daily_crime_counts['DayOfWeek'] >= 5).astype(int)

# Add rolling features
daily_crime_counts['RollingMean_7'] = daily_crime_counts['Crime_Count'].rolling(7).mean().fillna(
    daily_crime_counts['Crime_Count'].mean())
daily_crime_counts['RollingStd_7'] = daily_crime_counts['Crime_Count'].rolling(7).std().fillna(0)

# Lag Features
for lag in range(1, 8):
    daily_crime_counts[f'Lag_{lag}'] = daily_crime_counts['Crime_Count'].shift(lag)
daily_crime_counts.dropna(inplace=True)

# Prepare Data for Modeling
X = daily_crime_counts.drop(columns=['Crime_Count'])
y = daily_crime_counts['Crime_Count']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: XGBoost
xgb_model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='rmse'
)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_predictions = xgb_model.predict(X_test)

# Evaluation Metrics
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

print(f"XGBoost RMSE: {xgb_rmse}, MAE: {xgb_mae}, R²: {xgb_r2}")

# Plot Predictions vs Actuals
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(xgb_predictions, label='Predicted', marker='x', alpha=0.7)
plt.legend()
plt.title('XGBoost Predictions vs Actuals')
plt.savefig('xgboost_predictions_vs_actuals.png')  # Save the plot instead of showing it

# Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(X_train.columns, xgb_model.feature_importances_)
plt.title('Feature Importances')
plt.savefig('feature_importances.png')  # Save the plot instead of showing it

# Export Predictions
daily_crime_counts['Predicted_Crime_Count'] = xgb_model.predict(daily_crime_counts.drop(columns=['Crime_Count']))
daily_crime_counts[['Crime_Count', 'Predicted_Crime_Count']].to_csv("Crime_Predictions.csv")

# Print a sample of predictions
print("Sample Predictions:")
print(daily_crime_counts[['Crime_Count', 'Predicted_Crime_Count']].head())
