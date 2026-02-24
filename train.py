import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/crop_data.csv")

print("Original Dataset Shape:", df.shape)

# Remove missing values
df = df.dropna()

# Create Yield column (Target Variable)
df["Yield"] = df["Production"] / df["Area"]

# Remove unnecessary columns
df = df.drop(["Production", "District_Name", "Crop_Year"], axis=1)

print("After preprocessing Shape:", df.shape)

# One-hot encoding categorical variables
df = pd.get_dummies(df, columns=["Crop", "Season", "State_Name"])

print("After encoding Shape:", df.shape)

# Define Features and Target
X = df.drop("Yield", axis=1)
y = df["Yield"]

print("Splitting data...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Random Forest Model...")

# Random Forest with parallel processing
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Uses all CPU cores (faster)
)

model.fit(X_train, y_train)

print("Model training completed!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance:")
print("R2 Score:", r2)
print("MAE:", mae)

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and columns
pickle.dump(model, open("model/crop_model.pkl", "wb"))
pickle.dump(X.columns, open("model/model_columns.pkl", "wb"))

print("\nModel saved successfully in 'model' folder!")