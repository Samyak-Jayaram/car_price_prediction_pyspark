import datetime
import pandas as pd

# Load the CSV file into a DataFrame
dataset = pd.read_csv("cars_input.csv")

# Perform EDA and data cleaning
# Example: Count of cars based on manufacturers
dataset['Manufacturer'] = dataset['Name'].str.split(" ", expand=True)[0]

# Drop unnecessary columns
dataset.drop(["id","Name", "Location", "New_Price"], axis=1, inplace=True)

# Calculate age of cars
current_year = datetime.datetime.now().year
dataset['Year'] = current_year - dataset['Year']

# Convert Mileage to numeric
dataset['Mileage'] = pd.to_numeric(dataset['Mileage'].str.split(" ", expand=True)[0], errors='coerce')
dataset['Mileage'].fillna(dataset['Mileage'].mean(), inplace=True)

# Convert Engine and Power to numeric
dataset['Engine'] = pd.to_numeric(dataset['Engine'].str.split(" ", expand=True)[0], errors='coerce')
dataset['Power'] = pd.to_numeric(dataset['Power'].str.split(" ", expand=True)[0], errors='coerce')
dataset['Engine'].fillna(dataset['Engine'].mean(), inplace=True)
dataset['Power'].fillna(dataset['Power'].mean(), inplace=True)

# Fill missing values in Seats with mean
dataset['Seats'].fillna(dataset['Seats'].mean(), inplace=True)

# One-hot encode categorical variables
dataset = pd.get_dummies(dataset,
                         columns=["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first=True)

# Save cleaned dataset to a new CSV file
dataset.to_csv("cleaned_cars.csv", index=False)
