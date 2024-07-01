from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("CarPricePrediction").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Set log level to ERROR to reduce log noise

# Load the dataset
data = spark.read.csv("cleaned_cars_input.csv", header=True, inferSchema=True)

# Handle missing values
data = data.na.drop()

# Select the most important features based on the previous analysis
important_features = [
    "Power", "Engine", "Year", "Transmission_Manual", "Kilometers_Driven",
    "Fuel_Type_Diesel", "Fuel_Type_Petrol", "Seats", "Manufacturer_Mercedes-Benz",
    "Manufacturer_BMW", "Manufacturer_Audi"
]

# Create the vector assembler with important features
assembler = VectorAssembler(inputCols=important_features, outputCol="features")
data = assembler.transform(data)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2])

# Train the GBTRegressor model
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="Price", maxIter=10)

model = gbt.fit(train_data)

# Generate predictions
predictions = model.transform(test_data)

# Show actual prices and predictions
print("GBT Regressor predictions:")
predictions.select("Price", "prediction").show()

# Actual versus Predicted line plot
actual_vs_predicted = predictions.select("Price", "prediction").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(actual_vs_predicted['Price'], label='Actual')
plt.plot(actual_vs_predicted['prediction'], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# Actual versus Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(actual_vs_predicted['Price'], actual_vs_predicted['prediction'], alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices - Scatter Plot')
plt.show()

# Calculate R2
evaluator_r2 = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print(f"R²: {r2}")

# Evaluate the model
evaluator_rmse = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)
print(f"RMSE: {rmse}")

# Optimize the model (if needed)
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [5, 10, 15])
             .addGrid(gbt.maxIter, [10, 20, 30])
             .build())

crossval = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator_rmse,
                          numFolds=3)

cvModel = crossval.fit(train_data)

bestModel = cvModel.bestModel

# Evaluate best model
opt_predictions = bestModel.transform(test_data)


feature_importances = bestModel.stages[-1].featureImportances
for feature, importance in zip(important_features, feature_importances):
    print(f"{feature}: {importance}")

# Show actual prices and predictions
print("Optimized GBT Regressor predictions:")
opt_predictions.select("Price", "prediction").show()

# Actual versus Predicted line plot for optimized model
opt_actual_vs_predicted = opt_predictions.select("Price", "prediction").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(opt_actual_vs_predicted['Price'], label='Actual')
plt.plot(opt_actual_vs_predicted['prediction'], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Optimized Actual vs Predicted Prices')
plt.legend()
plt.show()

# Calculate R2 for optimized model
r2_opt = evaluator_r2.evaluate(opt_predictions)
print(f"Optimized R²: {r2_opt}")

# Correlation matrix
data_pd = data.select(important_features + ["Price"]).toPandas()
correlation_matrix = data_pd.corr()

plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.show()




# Save the best model
model_path = "./gbt_regressor_model_1"
bestModel.save(model_path)