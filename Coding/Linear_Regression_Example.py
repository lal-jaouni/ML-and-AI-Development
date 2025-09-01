import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
housing = fetch_california_housing(as_frame=True)

# Put into a DataFrame for easy inspection
df = housing.frame
print(df.head())


# Features (X) and target (y)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)



# Create the model
model = LinearRegression()

# Train (fit) the model on training data
model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R^2 :", r2)
print("MAE :", mae)
print("RMSE:", rmse)

coeffs = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("Intercept:", model.intercept_)
print(coeffs)
