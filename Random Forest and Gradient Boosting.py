import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('prophet_dataset_v1.csv')

# Assuming df is your DataFrame after loading the dataset
df['Datum'] = pd.to_datetime(df['Datum'])
df['Hour'] = df['Datum'].dt.hour
df['DayOfWeek'] = df['Datum'].dt.dayofweek
df['Month'] = df['Datum'].dt.month
df['TotalCost'] = df['Verbruik KwH'] * df['Prijs']

## Assuming df is your DataFrame
X = df[['Hour', 'DayOfWeek', 'Month', 'Verbruik KwH']]
y = df['TotalCost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# Evaluate Random Forest Model
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
rf_r2 = r2_score(y_test, rf_predictions)

# Evaluate Gradient Boosting Model
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = mean_squared_error(y_test, gb_predictions, squared=False)
gb_r2 = r2_score(y_test, gb_predictions)

print("Random Forest - MAE: {}, RMSE: {}, R²: {}".format(rf_mae, rf_rmse, rf_r2))
print("Gradient Boosting - MAE: {}, RMSE: {}, R²: {}".format(gb_mae, gb_rmse, gb_r2))

# Cross-validation
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)
cv_scores_gb = cross_val_score(gb_model, X, y, cv=5)

# Visualization of CV Scores
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.boxplot(cv_scores_rf)
plt.title('Random Forest CV Scores')
plt.ylabel('R² Score')

plt.subplot(1, 2, 2)
plt.boxplot(cv_scores_gb)
plt.title('Gradient Boosting CV Scores')
plt.ylabel('R² Score')

plt.show()

# Feature Importance Visualization for Random Forest
importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(10, 5))
plt.title('Feature Importances - Random Forest')
plt.bar(range(X.shape[1]), importances_rf[indices_rf], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices_rf], rotation=90)
plt.show()

# Selecting a single tree
# For Random Forest
single_tree_rf = rf_model.estimators_[0]
# For Gradient Boosting
single_tree_gb = gb_model.estimators_[0, 0]

# Plotting the tree
plt.figure(figsize=(20,10))
plot_tree(single_tree_rf, filled=True, feature_names=X.columns, max_depth=3)
plt.title("Single Tree from Random Forest")
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances in Random Forest')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()


