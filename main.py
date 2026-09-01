import pandas as pd

# Load the datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Check dataset size
print("Training shape:", train.shape)
print("Testing shape:", test.shape)

# Show column names
print("\nTraining columns:")
print(train.columns.tolist())

# Check data types and non-null values
print("\nTraining information:")
train.info()

# Check missing values
print("\nMissing values:")
print(train.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(train.duplicated().sum())

# Statistical summary
print("\nStatistical summary:")
print(train.describe())
print("\nUnique Building Types:")
print(train["Building Type"].unique())

print("\nBuilding Type Counts:")
print(train["Building Type"].value_counts())

print("\nDay of Week Values:")
print(train["Day of Week"].unique())

print("\nDay of Week Counts:")
print(train["Day of Week"].value_counts())

print("\nInvalid Square Footage:")
print((train["Square Footage"] <= 0).sum())

print("\nInvalid Number of Occupants:")
print((train["Number of Occupants"] <= 0).sum())

print("\nInvalid Appliances:")
print((train["Appliances Used"] <= 0).sum())

print("\nInvalid Temperature:")
print(((train["Average Temperature"] < -50) | 
       (train["Average Temperature"] > 60)).sum())

print("\nInvalid Energy Consumption:")
print((train["Energy Consumption"] <= 0).sum())
import matplotlib.pyplot as plt

# Energy Consumption by Building Type
building_energy = train.groupby("Building Type")["Energy Consumption"].mean()

print("\nAverage Energy Consumption by Building Type:")
print(building_energy)

building_energy.plot(kind="bar")

plt.title("Average Energy Consumption by Building Type")
plt.xlabel("Building Type")
plt.ylabel("Average Energy Consumption")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# Energy Consumption vs Square Footage

plt.figure(figsize=(8, 5))

plt.scatter(
    train["Square Footage"],
    train["Energy Consumption"]
)

plt.title("Energy Consumption vs Square Footage")
plt.xlabel("Square Footage")
plt.ylabel("Energy Consumption")

plt.tight_layout()
plt.show()
# Correlation with Energy Consumption

numeric_columns = [
    "Square Footage",
    "Number of Occupants",
    "Appliances Used",
    "Average Temperature",
    "Energy Consumption"
]

correlation = train[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation["Energy Consumption"].sort_values(ascending=False))
# Energy Consumption by Day Type

day_energy = train.groupby("Day of Week")["Energy Consumption"].agg(
    ["mean", "min", "max"]
)

print("\nEnergy Consumption by Day Type:")
print(day_energy)
# Energy Consumption by Building Type

building_energy = train.groupby("Building Type")["Energy Consumption"].agg(
    ["mean", "min", "max"]
).sort_values("mean", ascending=False)

print("\nEnergy Consumption by Building Type:")
print(building_energy)
# Energy Consumption vs Square Footage

print("\nEnergy Consumption vs Square Footage:")
print(
    train[["Square Footage", "Energy Consumption"]]
    .describe()
)
print("\nDataset Columns:")
print(train.columns.tolist())
# =========================
# Machine Learning
# =========================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Features
X = train.drop("Energy Consumption", axis=1)

# Target
y = train["Energy Consumption"]

# Categorical and numerical columns
categorical_features = ["Building Type", "Day of Week"]

numerical_features = [
    "Square Footage",
    "Number of Occupants",
    "Appliances Used",
    "Average Temperature"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
# =========================
# Linear Regression Model
# =========================

from sklearn.linear_model import LinearRegression

# Linear Regression model
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train Linear Regression
linear_model.fit(X_train, y_train)

# Prediction
linear_pred = linear_model.predict(X_test)

# Evaluation
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

print("\nLinear Regression Evaluation:")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R²:", linear_r2)


# =========================
# Model Comparison
# =========================

print("\nModel Comparison:")
print("---------------------------------------")
print(f"Random Forest   | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
print(f"Linear Regression | MAE: {linear_mae:.2f} | RMSE: {linear_rmse:.2f} | R²: {linear_r2:.4f}")
# =========================
# Cross-Validation
# =========================

from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    linear_model,
    X,
    y,
    cv=5,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
)

cv_mae = -cv_results["test_MAE"].mean()
cv_rmse = -cv_results["test_RMSE"].mean()
cv_r2 = cv_results["test_R2"].mean()

print("\n5-Fold Cross-Validation:")
print("Average MAE:", cv_mae)
print("Average RMSE:", cv_rmse)
print("Average R²:", cv_r2)
# =========================
# Actual vs Predicted
# =========================

comparison = pd.DataFrame({
    "Actual Energy Consumption": y_test.values,
    "Predicted Energy Consumption": linear_pred
})

print("\nActual vs Predicted:")
print(comparison.head(10))
# Prediction Error

comparison["Error"] = (
    comparison["Actual Energy Consumption"]
    - comparison["Predicted Energy Consumption"]
)

print("\nPrediction Error:")
print(comparison["Error"].describe())
# =========================
# Linear Regression Coefficients
# =========================

# Get feature names after preprocessing
feature_names = linear_model.named_steps["preprocessor"].get_feature_names_out()

# Get coefficients
coefficients = linear_model.named_steps["model"].coef_

# Create a DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Add absolute coefficient for ranking
feature_importance["Absolute Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

# Sort by importance
feature_importance = feature_importance.sort_values(
    "Absolute Coefficient",
    ascending=False
)

print("\nFeature Coefficients:")
print(feature_importance.to_string(index=False))
# =========================
# DSS Energy Consumption Prediction
# =========================

print("\n==============================")
print(" Energy Consumption DSS")
print("==============================")

# User inputs
building_type = input(
    "\nEnter Building Type (Residential / Commercial / Industrial): "
)

square_footage = float(
    input("Enter Square Footage: ")
)

occupants = int(
    input("Enter Number of Occupants: ")
)

appliances = int(
    input("Enter Number of Appliances Used: ")
)

temperature = float(
    input("Enter Average Temperature: ")
)

day_type = input(
    "Enter Day Type (Weekday / Weekend): "
)

# Create input DataFrame
new_building = pd.DataFrame({
    "Building Type": [building_type],
    "Square Footage": [square_footage],
    "Number of Occupants": [occupants],
    "Appliances Used": [appliances],
    "Average Temperature": [temperature],
    "Day of Week": [day_type]
})

# =========================
# AI Prediction & Decision Support
# =========================

predicted_energy = linear_model.predict(new_building)[0]

print("\n==============================")
print(" AI Prediction Result")
print("==============================")

print(
    f"Predicted Energy Consumption: "
    f"{predicted_energy:.2f}"
)

# AI Consumption Level and Recommendation
if predicted_energy >= 5000:
    consumption_level = "High"
    recommendation = (
        "Prioritize energy efficiency measures."
    )

elif predicted_energy >= 4000:
    consumption_level = "Moderate"
    recommendation = (
        "Consider optimization measures."
    )

else:
    consumption_level = "Lower"
    recommendation = (
        "Continue monitoring energy consumption."
    )

print("\nAI Consumption Level:")
print(consumption_level)

print("\nAI Recommendation:")
print(recommendation)


# =========================
# What-If Scenario
# =========================

what_if_building = new_building.copy()

# Reduce square footage by 10%
what_if_building["Square Footage"] = (
    what_if_building["Square Footage"] * 0.90
)

what_if_prediction = linear_model.predict(
    what_if_building
)[0]

estimated_reduction = (
    predicted_energy - what_if_prediction
)

print("\n==============================")
print(" What-If Scenario")
print("==============================")

print(
    f"Original Square Footage: "
    f"{square_footage:.0f}"
)

print(
    f"10% Reduced Square Footage: "
    f"{square_footage * 0.90:.0f}"
)

print(
    f"Original Prediction: "
    f"{predicted_energy:.2f}"
)

print(
    f"What-If Prediction: "
    f"{what_if_prediction:.2f}"
)

print(
    f"Estimated Reduction: "
    f"{estimated_reduction:.2f}"
)


# =========================
# Export AI Results for Power BI
# =========================

powerbi_data = X_test.copy()

powerbi_data["Actual Energy Consumption"] = y_test.values

powerbi_data["Predicted Energy Consumption"] = linear_pred

powerbi_data["Prediction Error"] = (
    powerbi_data["Actual Energy Consumption"]
    - powerbi_data["Predicted Energy Consumption"]
)


# AI Consumption Level
powerbi_data["Consumption Level"] = (
    powerbi_data["Predicted Energy Consumption"].apply(
        lambda x:
        "High" if x >= 5000
        else "Moderate" if x >= 4000
        else "Lower"
    )
)


# AI Recommendation
powerbi_data["Recommendation"] = (
    powerbi_data["Predicted Energy Consumption"].apply(
        lambda x:
        "Prioritize energy efficiency measures."
        if x >= 5000
        else
        "Consider optimization measures."
        if x >= 4000
        else
        "Continue monitoring energy consumption."
    )
)


# Save Power BI dataset
powerbi_data.to_csv(
    "energy_consumption_powerbi.csv",
    index=False
)

print("\n==============================")
print(" Power BI AI Dataset Created")
print("==============================")

print(
    "File: energy_consumption_powerbi.csv"
)

print(
    "Rows:",
    len(powerbi_data)
)

print(
    "Columns:",
    powerbi_data.columns.tolist()
)
