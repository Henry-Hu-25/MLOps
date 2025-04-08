import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the used car dataset
file_path = './data/used-cars_2cities_prep.csv'
cars = pd.read_csv(file_path)

# Drop unnecessary columns
cars.drop(columns=["pricestr", "lnprice", "lnodometer", "year", "name"], inplace=True)

# Define categorical columns
categorical_columns = ['area', 'subarea', 'condition', 'cylinders', 'drive', 
                       'fuel', 'paintcolor', 'size', 'transmission', 'type']

# One-hot encode categorical variables
cars_encoded = pd.get_dummies(cars, columns=categorical_columns, drop_first=True)

# Convert specific columns to boolean
cars_encoded["dealer"] = cars_encoded["dealer"].astype(bool)
cars_encoded["LE"] = cars_encoded["LE"].astype(bool)
cars_encoded["XLE"] = cars_encoded["XLE"].astype(bool)
cars_encoded["SE"] = cars_encoded["SE"].astype(bool)
cars_encoded["Hybrid"] = cars_encoded["Hybrid"].astype(bool)

# Split into features and target
X = cars_encoded.drop(columns="price")
y = cars_encoded["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# Combine features and target for saving
train_df = X_train.copy()
train_df["price"] = y_train

test_df = X_test.copy()
test_df["price"] = y_test

# Save to CSV
train_df.to_csv("./data/used_car_train_dcv.csv", index=False)
test_df.to_csv("./data/used_car_test_dcv.csv", index=False)
