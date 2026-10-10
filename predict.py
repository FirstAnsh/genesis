import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/rf_Species.joblib")

# Print the exact feature names the model expects
print("Expected features:", list(model.feature_names_in_))

# Create a single sample matching the expected columns
# Provide a dummy value (e.g., 1) for the Id/index column if present
sample_data = {col: [0] for col in model.feature_names_in_}

# Fill in typical measurement values where matching
for col in model.feature_names_in_:
    col_lower = col.lower()
    if "sepallength" in col_lower:
        sample_data[col] = [5.1]
    elif "sepalwidth" in col_lower:
        sample_data[col] = [3.5]
    elif "petallength" in col_lower:
        sample_data[col] = [1.4]
    elif "petalwidth" in col_lower:
        sample_data[col] = [0.2]
    elif "id" in col_lower:
        sample_data[col] = [1]

df_sample = pd.DataFrame(sample_data)

prediction = model.predict(df_sample)
print(f"Predicted Class: {prediction[0]}")