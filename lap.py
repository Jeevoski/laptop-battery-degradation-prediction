import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. GENERATE MOCK DATA
# ==========================================
np.random.seed(42)
cycles = np.arange(1, 501)
# Battery health dropping from ~100 to ~70 over 500 cycles
soh = 100 - (cycles * 0.06) + np.random.normal(0, 2, 500) 

# Convert to NumPy arrays and reshape for math operations
X = cycles.reshape(-1, 1) 
y = soh.reshape(-1, 1)

# Feature Scaling: CRITICAL for Gradient Descent. 
# Without scaling cycles (1-500), the gradients explode and the model fails.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. HYPERPARAMETERS (The dials you set)
# ==========================================
learning_rate = 0.1  # How big of a step the model takes down the error hill
epochs = 50          # How many times the model loops through the data

# ==========================================
# 3. INITIALIZE PARAMETERS (w and b)
# ==========================================
# We start with a random guess for our line: y = wx + b
w = np.random.randn(1, 1) # Weight (Slope)
b = np.random.randn(1, 1) # Bias (Intercept)

# List to keep track of the error over time so we can graph it later
loss_history = []

print("Starting Training...")

# ==========================================
# 4. GRADIENT DESCENT LOOP
# ==========================================
for epoch in range(epochs):
    # Step A: Make a prediction with current w and b
    y_pred = np.dot(X_scaled, w) + b
    
    # Step B: Calculate LOSS (Mean Squared Error)
    # How wrong is the current line?
    error = y_pred - y
    mse = np.mean(error ** 2)
    loss_history.append(mse)
    
    # Step C: Calculate Gradients
    # The calculus part: figuring out which direction to adjust w and b to reduce error
    dw = (2 / len(X_scaled)) * np.dot(X_scaled.T, error)
    db = (2 / len(X_scaled)) * np.sum(error)
    
    # Step D: Update Parameters
    # Move w and b in the opposite direction of the gradient
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss (MSE) = {mse:.2f}")

print(f"Training Complete. Final Loss: {loss_history[-1]:.2f}")

# ==========================================
# 5. PLOTTING THE RESULTS
# ==========================================
# We create a figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Graph 1: The Gradient Descent Loss Curve ---
ax1.plot(range(epochs), loss_history, color='red', marker='o')
ax1.set_title('Loss (MSE) over Epochs')
ax1.set_xlabel('Epochs (Iterations)')
ax1.set_ylabel('Mean Squared Error')
ax1.grid(True)

# --- Graph 2: The Final Best Fit Line ---
# We use the final trained 'w' and 'b' to draw our prediction line
final_predictions = np.dot(X_scaled, w) + b

ax2.scatter(X, y, color='blue', alpha=0.5, label='Actual Battery Data', s=10)
ax2.plot(X, final_predictions, color='green', linewidth=3, label='Learned Best Fit Line')
ax2.set_title('Linear Regression: Cycle Count vs State of Health')
ax2.set_xlabel('Cycle Count')
ax2.set_ylabel('State of Health (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()