from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd

df = pd.read_csv("data/crop_data.csv")
df = df.dropna()
df["Yield"] = df["Production"] / df["Area"]

df_encoded = pd.get_dummies(df, columns=["Crop", "Season", "State_Name"])

X = df_encoded.drop(["Yield", "Production", "District_Name"], axis=1, errors="ignore")
y = df_encoded["Yield"]

# SMALLER MODEL
model = RandomForestRegressor(
    n_estimators=20,      # ↓ Reduced trees
    max_depth=10,         # ↓ Limit depth
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

with open("crop_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("Model saved successfully")