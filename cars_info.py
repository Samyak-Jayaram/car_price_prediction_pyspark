import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('./cleaned_cars_input.csv')

# Print the number of rows and columns
num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Print the datatype of each column
print("\nColumn Data Types:")
print(df.dtypes)
