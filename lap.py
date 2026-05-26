import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Generating Mock Data (Simulating 500 charge cycles of a laptop)
np.random.seed(42)
cycles = np.arange(1, 501)
# Temperature generally increases slightly as battery ages
temperature = np.random.normal(35, 5, 500) + (cycles * 0.02) 
# Voltage drops slightly as it ages
voltage = np.random.normal(11.5, 0.5, 500) - (cycles * 0.001)
# State of Health (Target) degrades from 100% down to ~60% non-linearly
soh = 100 - (cycles ** 1.2) * 0.02 + np.random.normal(0, 1, 500)

df = pd.DataFrame({
    'Cycle_Count': cycles,
    'Temperature_C': temperature,
    'Voltage_V': voltage,
    'State_of_Health': soh
})

# Introduce some fake missing values to simulate sensor failure
df.loc[10:15, 'Temperature_C'] = np.nan
print("Initial Data Head:\n", df.head())

# 2. Preprocessing: Handle Missing Values using Forward Fill
df['Temperature_C'] = df['Temperature_C'].ffill()

# 3. Feature Engineering: Create a rolling average for temperature
# (Simulating prolonged heat exposure which degrades batteries faster)
df['Temp_Rolling_Avg_5'] = df['Temperature_C'].rolling(window=5, min_periods=1).mean()

# Define our Features (X) and Target (y)
X = df[['Cycle_Count', 'Temperature_C', 'Voltage_V', 'Temp_Rolling_Avg_5']]
y = df['State_of_Health']

# Split data: 80% for training the model, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Initialize the Scaler
scaler = StandardScaler()

# Fit on training data AND transform it. 
# (We only 'transform' test data to prevent data leakage from the future)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the unseen test data
predictions = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}%")

# Quick look at an actual vs predicted value
comparison = pd.DataFrame({'Actual_SoH': y_test.values[:5], 'Predicted_SoH': predictions[:5]})
print("\nPrediction Check:\n", comparison)