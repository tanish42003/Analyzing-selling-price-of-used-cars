import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('car data.csv')

# Drop unnecessary columns
df.drop(columns=['Car_Name'], inplace=True)

# Convert year to car age
df['Car_Age'] = 2025 - df['Year']
df.drop(columns=['Year'], inplace=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Define features and target variable
X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# GUI Function for Prediction
def predict_price():
    try:
        present_price = float(present_price_entry.get())
        kms_driven = int(kms_driven_entry.get())
        owner = int(owner_entry.get())
        car_age = int(car_age_entry.get())
        fuel_type = fuel_var.get()
        seller_type = seller_var.get()
        transmission = trans_var.get()

        # Create input array
        input_data = np.array([[present_price, kms_driven, owner, car_age,
                                1 if fuel_type == "Diesel" else 0,
                                1 if fuel_type == "Petrol" else 0,
                                1 if seller_type == "Individual" else 0,
                                1 if transmission == "Manual" else 0]])

        prediction = model.predict(input_data)[0]
        result_label.config(text=f"Predicted Price: ₹{prediction:.2f}")
    except ValueError:
        result_label.config(text="Invalid input. Please enter valid values.")


# GUI Setup
root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("400x400")

tk.Label(root, text="Present Price (Lakhs):").pack()
present_price_entry = tk.Entry(root)
present_price_entry.pack()

tk.Label(root, text="Kilometers Driven:").pack()
kms_driven_entry = tk.Entry(root)
kms_driven_entry.pack()

tk.Label(root, text="Number of Owners (0/1/2+):").pack()
owner_entry = tk.Entry(root)
owner_entry.pack()

tk.Label(root, text="Car Age (Years):").pack()
car_age_entry = tk.Entry(root)
car_age_entry.pack()

fuel_var = tk.StringVar(value="Petrol")
tk.Label(root, text="Fuel Type:").pack()
ttk.Combobox(root, textvariable=fuel_var, values=["Petrol", "Diesel", "CNG"]).pack()

seller_var = tk.StringVar(value="Dealer")
tk.Label(root, text="Seller Type:").pack()
ttk.Combobox(root, textvariable=seller_var, values=["Dealer", "Individual"]).pack()

trans_var = tk.StringVar(value="Manual")
tk.Label(root, text="Transmission:").pack()
ttk.Combobox(root, textvariable=trans_var, values=["Manual", "Automatic"]).pack()

predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
