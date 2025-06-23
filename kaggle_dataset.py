
import os
import pandas as pd
from  kaggle_data_loader import KaggleDataLoader

class KaggleDataSet:
    POLARITY_COLUMN_NAME = "Label"
    TITLE_COLUMN_NAME = "Title"
    REVIEW_COLUMN_NAME = "Review"
    POLARITY_VALUE_1 = 1
    POLARITY_VALUE_2 = 2
    
    def __init__(self, kaggle_data_loader: KaggleDataLoader):
        self.kaggle_data_loader = kaggle_data_loader
        self.train_df = kaggle_data_loader.get_train_df()
        self.test_df = kaggle_data_loader.get_test_df()

    def get_train_df(self):
        return self.train_df
    
    def get_test_df(self):
        return self.test_df
    
# Example usage:
if __name__ == "__main__":
    print("KaggleDataSet")