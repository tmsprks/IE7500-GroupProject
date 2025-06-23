
import os
import pandas as pd
import kagglehub

class SADataLoader:
    
    def __init__(self, path_to_train_csv_file: str = None, path_to_test_csv_file: str = None, path_to_validation_csv_file: str = None):
        self.path_to_train_csv_file = path_to_train_csv_file
        self.path_to_test_csv_file = path_to_test_csv_file
        self.path_to_validation_csv_file = path_to_validation_csv_file
        self.train_df = None
        self.test_df = None
        self.validation_df = None

    def __str__(self):
        return f"Train csv: {self.path_to_train_csv_file}, size: {len(self.get_train_df())}, Test csv: {self.path_to_test_csv_file}, size: {len(self.get_test_df())}, Test csv: {self.path_to_validation_csv_file}, size: {len(self.get_validation_df())}"

    def load_data(self, column_names: list[str]):
        print("Path to train csv file:", self.path_to_train_csv_file)
        self.train_df = pd.read_csv(self.path_to_train_csv_file, header=None, names=column_names)

        print("Path to test csv file:", self.path_to_test_csv_file)
        self.test_df = pd.read_csv(self.path_to_test_csv_file, header=None, names=column_names)

        if self.path_to_validation_csv_file is not None:
            print("Path to validation csv file:", self.path_to_validation_csv_file)
            self.validation_df = pd.read_csv(self.path_to_validation_csv_file, header=None, names=column_names)

        print(f"Train size: {len(self.train_df)}")
        print(f"Test size: {len(self.test_df)}")

        if self.validation_df is not None:
            print(f"Validation size: {len(self.validation_df)}")
        else:
            print("Validation size: 0")

        return (self.path_to_train_csv_file, self.path_to_test_csv_file, self.path_to_validation_csv_file)
    
    def get_train_df(self):
        return self.train_df
    
    def get_test_df(self):
        return self.test_df
    
    def get_validation_df(self):
        return self.validation_df


