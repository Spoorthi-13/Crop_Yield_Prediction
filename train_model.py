import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load data
df_original = pd.read_csv("data/crop_data.csv")
df_original = df_original.dropna()
df_original["Yield"] = df_original["Production"] / df_original["Area"]

# One-hot encode
df_encoded = pd.get_dummies(
    df_original,
    columns=["Crop", "Season", "State_Name"]
)

X = df_encoded.drop(
    ["Yield", "Production", "District_Name"],
    axis=1,
    errors="ignore"
)

y = df_encoded["Yield"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and columns
with open("crop_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("Model saved successfully.")