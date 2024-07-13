import datetime
import pandas as pd


dataset = pd.read_csv("cars_input.csv")

dataset['Manufacturer'] = dataset['Name'].str.split(" ", expand=True)[0]


dataset.drop(["id","Name", "Location", "New_Price"], axis=1, inplace=True)


current_year = datetime.datetime.now().year
dataset['Year'] = current_year - dataset['Year']


dataset['Mileage'] = pd.to_numeric(dataset['Mileage'].str.split(" ", expand=True)[0], errors='coerce')
dataset['Mileage'].fillna(dataset['Mileage'].mean(), inplace=True)


dataset['Engine'] = pd.to_numeric(dataset['Engine'].str.split(" ", expand=True)[0], errors='coerce')
dataset['Power'] = pd.to_numeric(dataset['Power'].str.split(" ", expand=True)[0], errors='coerce')
dataset['Engine'].fillna(dataset['Engine'].mean(), inplace=True)
dataset['Power'].fillna(dataset['Power'].mean(), inplace=True)

dataset['Seats'].fillna(dataset['Seats'].mean(), inplace=True)


dataset = pd.get_dummies(dataset,
                         columns=["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first=True)


dataset.to_csv("cleaned_cars.csv", index=False)
