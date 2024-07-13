import tkinter as tk
from tkinter import PhotoImage, ttk, messagebox
import traceback
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.feature import VectorAssembler,StandardScalerModel
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import matplotlib.pyplot as plt


spark = SparkSession.builder.appName("CarPricePrediction").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


model_path = "./final_gbt_regressor_model"
loaded_model = GBTRegressionModel.load(model_path)

scaler_model = StandardScalerModel.load("scaler_model")


class CarPricePredictionGUI:


    def __init__(self, master):
        self.master = master
        master.title("Used Car Price Prediction")
        master.geometry("1000x700")


        self.important_features = [
                "Power", "Engine", "Year", "Transmission_Manual", "Kilometers_Driven",
                "Fuel_Type_Diesel", "Fuel_Type_Petrol", "Seats", "Manufacturer_Mercedes-Benz",
                "Manufacturer_BMW", "Manufacturer_Audi"
            ]

        title_label = tk.Label(self.master, text="Used Car Price Prediction", font=('Times New Roman', 30, 'bold'))
        title_label.grid(row=0, column=4, columnspan=6, pady=20)


        self.create_input_fields()
        self.create_checkboxes()
        self.create_predict_button()
        self.create_load_button()
        self.create_r2_score_button()
        self.line_plot_button()
        self.scatter_plot_button()
        self.cor_mat_button()
        self.fea_imp_button()

        self.initialize_variables()
       

    def initialize_variables(self):
        self.Year = 0
        self.Kilometers_Driven = 0
        self.Mileage = 0
        self.Engine = 0
        self.Power = 0
        self.Seats = 0

        manufacturers = ["Audi", "BMW", "Bentley", "Chevrolet", "Datsun", "Fiat", "Force", "Ford", "Honda", "Hyundai", "ISUZU", "Isuzu", "Jaguar", "Jeep", "Lamborghini", "Land", "Mahindra", "Maruti", "Mercedes-Benz", "Mini", "Mitsubishi", "Nissan", "Porsche", "Renault", "Skoda", "Smart", "Tata", "Toyota", "Volkswagen", "Volvo"]
        fuel_types = ["Diesel", "Electric", "LPG", "Petrol"]
        transmission_types = ["Manual"]
        owner_types = ["Fourth & Above", "Second", "Third"]

        for manu in manufacturers:
            setattr(self, f"Manufacturer_{manu}", False)
        for fuel in fuel_types:
            setattr(self, f"Fuel_Type_{fuel}", False)
        for trans in transmission_types:
            setattr(self, f"Transmission_{trans}", False)
        for owner in owner_types:
            setattr(self, f"Owner_Type_{owner}", False)

    def create_input_fields(self):
        input_fields = ["Year", "Kilometers driven", "Mileage", "Engine", "Power", "Seats"]
        self.input_vars = {}

        for i, field in enumerate(input_fields):
            label = tk.Label(self.master, text=field,font=('Times New Roman', 18))
            label.grid(row=i+1, column=4, padx=5, pady=5, sticky="e")

            var = tk.StringVar()
            var.trace("w", lambda name, index, mode, field=field, var=var: self.update_input_variable(field, var))
            entry = tk.Entry(self.master, textvariable=var,bd=2,font=('Times New Roman', 14))
            entry.grid(row=i+1, column=5, padx=5, pady=5)

            self.input_vars[field] = var

    def update_input_variable(self, field, var):
        try:
            value = float(var.get())
            field = field.replace(" ", "_") 
            if(value==0):
                self.result_label.config(text=f"Error:{field} must not be zero")
                return
            else:
                setattr(self, field, value)
        except ValueError:
            setattr(self, field, 0)
            self.result_label.config(text=f"Error: {field} must be a number.")
            self.master.after(30000, self.clear_result_label)


    def create_checkboxes(self):
        self.manufacturers = ["Audi", "BMW", "Bentley", "Chevrolet", "Datsun", "Fiat", "Force", "Ford", "Honda", "Hyundai", "ISUZU", "Isuzu", "Jaguar", "Jeep", "Lamborghini", "Land", "Mahindra", "Maruti", "Mercedes-Benz", "Mini", "Mitsubishi", "Nissan", "Porsche", "Renault", "Skoda", "Smart", "Tata", "Toyota", "Volkswagen", "Volvo"]
        self.fuel_types = ["Diesel", "Electric", "LPG", "Petrol"]
        transmission_types = ["Manual"]
        self.owner_types = ["Fourth & Above", "Second", "Third"]

        self.checkbox_vars = {}


        manu_frame = tk.LabelFrame(self.master, text="Manufacturer",font=('Times New Roman', 18,'bold'))
        manu_frame.grid(row=1, column=8, rowspan=10, padx=10, pady=10, sticky="n")

        for i, manu in enumerate(self.manufacturers):
            var = tk.BooleanVar()
            var.trace("w", lambda name, index, mode, field=f"Manufacturer_{manu}", var=var: self.update_checkbox_variable(field, var))
            cb = tk.Checkbutton(manu_frame, text=manu, variable=var,font=('Times New Roman', 18))
            cb.grid(row=i%15, column=i//15, sticky="w")
            self.checkbox_vars[f"Manufacturer_{manu}"] = var


        fuel_frame = tk.LabelFrame(self.master, text="Fuel Type",font=('Times New Roman', 18,'bold'))
        fuel_frame.grid(row=2, column=9, padx=10, pady=10, sticky="w")

        for i, fuel in enumerate(self.fuel_types):
            var = tk.BooleanVar()
            var.trace("w", lambda name, index, mode, field=f"Fuel_Type_{fuel}", var=var: self.update_checkbox_variable(field, var))
            cb = tk.Checkbutton(fuel_frame, text=fuel, variable=var,font=('Times New Roman', 18))
            cb.grid(row=0, column=i, padx=5, pady=5)
            self.checkbox_vars[f"Fuel_Type_{fuel}"] = var

        trans_frame = tk.LabelFrame(self.master, text="Transmission",font=('Times New Roman', 18,'bold'))
        trans_frame.grid(row=3, column=9, padx=10, pady=10, sticky="w")

        for i, trans in enumerate(transmission_types):
            var = tk.BooleanVar()
            var.trace("w", lambda name, index, mode, field=f"Transmission_{trans}", var=var: self.update_checkbox_variable(field, var))
            cb = tk.Checkbutton(trans_frame, text=trans, variable=var,font=('Times New Roman', 18))
            cb.grid(row=0, column=i, padx=5, pady=5)
            self.checkbox_vars[f"Transmission_{trans}"] = var


        owner_frame = tk.LabelFrame(self.master, text="Owner Type",font=('Times New Roman', 18,'bold'))
        owner_frame.grid(row=4, column=9, padx=10, pady=10, sticky="w")

        for i, owner in enumerate(self.owner_types):
            var = tk.BooleanVar()
            var.trace("w", lambda name, index, mode, field=f"Owner_Type_{owner}", var=var: self.update_checkbox_variable(field, var))
            cb = tk.Checkbutton(owner_frame, text=owner, variable=var,font=('Times New Roman', 18))
            cb.grid(row=0, column=i, padx=5, pady=5)
            self.checkbox_vars[f"Owner_Type_{owner}"] = var

    def update_checkbox_variable(self, field, var):
        setattr(self, field, bool(var.get()))

    def create_predict_button(self):
        predict_button = tk.Button(self.master, text="Predict Price", font=('Times New Roman', 18, 'bold'),command=self.create_input_csv)
        predict_button.grid(row=7, column=5, pady=10)

        self.result_label = tk.Label(self.master, text="",font=('Times New Roman', 14, 'bold'))
        self.result_label.grid(row=15, column=5, pady=10)

    def create_load_button(self):
        load_button = tk.Button(self.master, text="Predict Top 20 rows",font=('Times New Roman', 14, 'bold'), command=self.load_and_predict)
        load_button.grid(row=1, column=1, pady=10)

    def create_r2_score_button(self):
        r2_score_button = tk.Button(self.master, text="Check Accuracy", font=('Times New Roman', 14, 'bold'),command=self.r2_score_calculate)
        r2_score_button.grid(row=2, column=1, pady=10)

    def cor_mat_button(self):
        corr_mat_button = tk.Button(self.master, text="Display Correlation matrix", font=('Times New Roman', 14, 'bold'),command=self.corr_matrix)
        corr_mat_button.grid(row=3, column=1, pady=10)

    def line_plot_button(self):
        line_plot_button = tk.Button(self.master, text="Display line plot", font=('Times New Roman', 14, 'bold'),command=self.create_line_plot)
        line_plot_button.grid(row=4, column=1, pady=10)


    def scatter_plot_button(self):
        scatter_plot_button = tk.Button(self.master, text="Display scatter plot", font=('Times New Roman', 14, 'bold'),command=self.create_scatter_plot)
        scatter_plot_button.grid(row=5, column=1, pady=10)

    def fea_imp_button(self):
        fea_imp_button = tk.Button(self.master, text="Display feature importances",font=('Times New Roman', 14, 'bold'), command=self.display_feature_importance)
        fea_imp_button.grid(row=6, column=1, pady=10) 


    def predict_price(self):
        try:
                   
            manufacturers_checked = sum(getattr(self, f"Manufacturer_{manu}") for manu in self.manufacturers)
            if manufacturers_checked != 1:
                self.result_label.config(text="Please select exactly one manufacturer.")
                return


            fuel_types_checked = sum(getattr(self, f"Fuel_Type_{fuel}") for fuel in self.fuel_types)
            if fuel_types_checked != 1:
                self.result_label.config(text="Please select exactly one fuel type.")
                return


            owner_types_checked = sum(getattr(self, f"Owner_Type_{owner}") for owner in self.owner_types)
            if owner_types_checked not in (0, 1):
                self.result_label.config(text="Please select exactly one  or zero owner type.")
                return

            
            input_df = spark.read.csv("input.csv", header=True, inferSchema=True)

            assembler = VectorAssembler(inputCols=self.important_features, outputCol="features")
            input_df = assembler.transform(input_df)

            input_df = scaler_model.transform(input_df)

            prediction = loaded_model.transform(input_df)

            predicted_price = prediction.select("prediction").first()[0]

            self.result_label.config(text=f"Predicted Price: {predicted_price:.2f} lakh rupees")

            self.master.after(30000, self.clear_result_label)

        except Exception as e:
            error_message = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_message)
            self.result_label.config(text="Incorrect values entered.")



    def clear_result_label(self):
        self.result_label.config(text="")

    def create_input_csv(self):
        input_data = {}


        input_fields = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
        for field in input_fields:
            input_data[field] = getattr(self, field)


        manufacturers = ["Audi", "BMW", "Bentley", "Chevrolet", "Datsun", "Fiat", "Force", "Ford", "Honda", "Hyundai", "ISUZU", "Isuzu", "Jaguar", "Jeep", "Lamborghini", "Land", "Mahindra", "Maruti", "Mercedes-Benz", "Mini", "Mitsubishi", "Nissan", "Porsche", "Renault", "Skoda", "Smart", "Tata", "Toyota", "Volkswagen", "Volvo"]
        fuel_types = ["Diesel", "Electric", "LPG", "Petrol"]
        transmission_types = ["Manual"]
        owner_types = ["Fourth & Above", "Second", "Third"]

        for manu in manufacturers:
            input_data[f"Manufacturer_{manu}"] = getattr(self, f"Manufacturer_{manu}")
        for fuel in fuel_types:
            input_data[f"Fuel_Type_{fuel}"] = getattr(self, f"Fuel_Type_{fuel}")
        for trans in transmission_types:
            input_data[f"Transmission_{trans}"] = getattr(self, f"Transmission_{trans}")
        for owner in owner_types:
            input_data[f"Owner_Type_{owner}"] = getattr(self, f"Owner_Type_{owner}")

        df = pd.DataFrame([input_data])
        df.to_csv("input.csv", index=False)

        self.predict_price()

       


    def load_and_predict(self):
        spark_df_load = spark.read.csv("cleaned_cars_input.csv", header=True, inferSchema=True)
        spark_df_load = spark_df_load.na.drop()

        assembler = VectorAssembler(inputCols=self.important_features, outputCol="features")
        spark_df_load = assembler.transform(spark_df_load)

        spark_df_load = scaler_model.transform(spark_df_load)
        train_data, test_data = spark_df_load.randomSplit([0.8, 0.2])

        predictions = loaded_model.transform(test_data)
        top_predictions = predictions.select("Price", "prediction").limit(20).collect()
        
        self.data_pd = spark_df_load.select(self.important_features + ["Price"]).toPandas()
        self.actual_vs_predicted=predictions.select("Price", "prediction").toPandas()

        if top_predictions:
            table_window = tk.Toplevel(self.master)
            table_window.title("Top 20 Predictions")

            # Define style
            style = ttk.Style()
            style.configure("Treeview", font=("Times New Roman", 18))
            style.configure("Treeview.Heading", font=("Times New Roman", 18, "bold"))
            style.configure("Treeview", rowheight=40)

            table = ttk.Treeview(table_window, columns=("Actual Price", "Predicted Price"), show="headings")
            table.heading("Actual Price", text="Actual Price (lakh rupees)")
            table.heading("Predicted Price", text="Predicted Price (lakh rupees)")
            table.pack(fill=tk.BOTH, expand=True)

        for row in top_predictions:
            actual_price = row['Price']
            predicted_price = row['prediction']
            table.insert("", "end", values=(f"\t\t\t{actual_price:.2f}", f"\t\t\t{predicted_price:.2f}"))


    

    def create_line_plot(self):
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.actual_vs_predicted['Price'], label='Actual')
            plt.plot(self.actual_vs_predicted['prediction'], label='Predicted')
            plt.xlabel('Index')
            plt.ylabel('Price')
            plt.title('Actual vs Predicted Prices')
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showinfo("Error", "Error generating line plot") 

    def create_scatter_plot(self):
        try:

            plt.figure(figsize=(8, 6))
            plt.scatter(self.actual_vs_predicted['Price'], self.actual_vs_predicted['prediction'], alpha=0.5)
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title('Actual vs Predicted Prices - Scatter Plot')
            plt.show()
        except Exception as e:
            messagebox.showinfo("Error", "Error generating scatter plot") 



    def r2_score_calculate(self):
        try:

            data = spark.read.csv("cleaned_cars_input.csv", header=True, inferSchema=True)

            data = data.na.drop()

            assembler = VectorAssembler(inputCols=self.important_features, outputCol="features")
            data = assembler.transform(data)
            data = scaler_model.transform(data)

            train_data, test_data = data.randomSplit([0.8, 0.2])
            predictions = loaded_model.transform(test_data)


            evaluator_r2 = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="r2")
            r2 = evaluator_r2.evaluate(predictions)

            evaluator_rmse = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
            rmse = evaluator_rmse.evaluate(predictions)

            self.show_custom_message(f"R²: {r2:.4f}\nRMSE: {rmse:.4f}")

        except Exception as e:
            self.show_custom_message(f"An error occurred during model evaluation: {str(e)}", error=True)

    def show_custom_message(self, message, error=False):

        top = tk.Toplevel(self.master)
        top.title("Model Evaluation" if not error else "Evaluation Error")
        top.geometry("400x200")

        label = tk.Label(top, text=message, font=("Times New Roman", 18), wraplength=380)
        label.pack(padx=10, pady=10)

        button = tk.Button(top, text="OK", command=top.destroy, font=("Times New Roman", 14))
        button.pack(pady=10)

    def display_feature_importance(self):
        importances = loaded_model.featureImportances
        feature_importances = pd.DataFrame(importances.toArray(), index=self.important_features, columns=["Importance"]).sort_values(by="Importance", ascending=False)

        table_window = tk.Toplevel(self.master)
        table_window.title("Feature importance")

        style = ttk.Style()
        style.configure("Treeview", font=("Times New Roman", 18))
        style.configure("Treeview.Heading", font=("Times New Roman", 18, "bold"))
        style.configure("Treeview", rowheight=40)

        table = ttk.Treeview(table_window, columns=("Field", "Importance"), show="headings")
        table.heading("Field", text="Parameter")
        table.heading("Importance", text="Importance")
        table.pack(fill=tk.BOTH, expand=True)

        for index, row in feature_importances.iterrows():
            parameter = index
            importance = row['Importance']
            table.insert("", "end", values=(parameter, f"{importance:.4f}"))


    def corr_matrix(self):
        try:
            correlation_matrix = self.data_pd.corr()

            plt.figure(figsize=(10, 8))
            plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
            plt.colorbar()
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
            plt.title('Correlation Matrix')
            plt.show()

        except Exception as e:
            messagebox.showinfo("Error", "Error generating correlation matrix")



if __name__ == "__main__":
    root = tk.Tk()
    gui = CarPricePredictionGUI(root)
    root.mainloop()
