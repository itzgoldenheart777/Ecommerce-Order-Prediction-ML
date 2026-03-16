import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data/Meesho Orders Aug.csv")

print("Columns in dataset:")
print(data.columns)

# Rename columns correctly
data.rename(columns={
    "Reason for Credit Entry": "Order_Status",
    "Supplier Listed Price (Incl. GST + Commission)": "Price"
}, inplace=True)

# Convert date column
data["Order Date"] = pd.to_datetime(data["Order Date"])

# Feature engineering
data["Month"] = data["Order Date"].dt.month

# Remove missing values
data = data.dropna(subset=["Customer State","Quantity","Price","Order_Status"])

# Encoding categorical data
le_state = LabelEncoder()
le_status = LabelEncoder()

data["Customer State"] = le_state.fit_transform(data["Customer State"])
data["Order_Status"] = le_status.fit_transform(data["Order_Status"])

# Features and target
X = data[["Customer State","Quantity","Price","Month"]]
y = data["Order_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
