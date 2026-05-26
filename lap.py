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