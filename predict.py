import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

print("=== E-Commerce Order Prediction System ===")
print("Type 'q' anytime to exit.\n")

# State mapping (example encoding)
state_map = {
    "maharashtra": 0,
    "delhi": 1,
    "gujarat": 2,
    "karnataka": 3
}

status_map = {
    0: "Cancelled",
    1: "Delivered",
    2: "RTO"
}

while True:

    state_input = input("Enter Customer State (Maharashtra/Delhi/Gujarat/Karnataka): ")

    if state_input.lower() == "q":
        print("Exiting system...")
        break

    state_encoded = state_map.get(state_input.lower())

    if state_encoded is None:
        print("Invalid state. Try again.\n")
        continue

    qty = int(input("Enter Quantity: "))
    price = float(input("Enter Product Price: "))
    month = int(input("Enter Order Month (1-12): "))

    # Create dataframe
    sample = pd.DataFrame({
        "Customer State": [state_encoded],
        "Quantity": [qty],
        "Price": [price],
        "Month": [month]
    })

    # Predict
    prediction = model.predict(sample)

    print("\nPredicted Order Status:", status_map.get(prediction[0]))
    print("-" * 40)
