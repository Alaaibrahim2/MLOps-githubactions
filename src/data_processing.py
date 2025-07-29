import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os

print("Script started!")  # لتأكيد التشغيل

class DataProcessing:
    def __init__(self, file_path):
        print("Initializing DataProcessing...")
        self.file_path = file_path
        self.df = None
        self.processed_data_path = "artifacts/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.file_path)

    def handle_outliers(self, column):
        print("Handling outliers...")
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        median = np.median(self.df[column])
        self.df[column] = self.df[column].apply(lambda x: median if x < lower or x > upper else x)

    def split_data(self):
        print("Splitting and saving data...")
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        joblib.dump(X_train, os.path.join(self.processed_data_path, "X_train.pkl"))
        joblib.dump(X_test, os.path.join(self.processed_data_path, "X_test.pkl"))
        joblib.dump(y_train, os.path.join(self.processed_data_path, "y_train.pkl"))
        joblib.dump(y_test, os.path.join(self.processed_data_path, "y_test.pkl"))

    def run(self):
        print("Running pipeline...")
        self.load_data()
        self.handle_outliers("SepalWidthCm")
        self.split_data()

if __name__ == "__main__":
    print("Main block running...")
    data_processor = DataProcessing("artifacts/raw/data.csv")
    data_processor.run()
