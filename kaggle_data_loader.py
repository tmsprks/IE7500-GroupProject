
import os
from typing import List
import pandas as pd
import kagglehub

class KaggleDataLoader:
    
    def __init__(self, kaggle_path: str):
        self.kaggle_path = kaggle_path
        self.train_df = None
        self.test_df = None

    def load_data(self, column_names: List[str]) -> str:
        # Download latest version
        path = kagglehub.dataset_download(self.kaggle_path)

        print("Path to dataset files:", path)
        print(path)
        print(os.listdir(path))

        train_path = path+"\\train.csv"
        test_path = path+"\\test.csv"

        self.train_df = pd.read_csv(train_path, header=None, names=column_names)
        self.test_df = pd.read_csv(test_path, header=None, names=column_names)

        return path
    
    def get_train_df(self) -> pd.DataFrame:
        return self.train_df
    
    def get_test_df(self) -> pd.DataFrame:
        return self.test_df
    
# Example usage:
if __name__ == "__main__":
    kaggle_path = "kritanjalijain/amazon-reviews"

    kaggle_data_loader = KaggleDataLoader(kaggle_path)
    column_names = ["Label", "Title", "Review"]
    
    kaggle_data_loader.load_data(column_names)
    print(kaggle_data_loader.get_train_df().head(5))
    print(kaggle_data_loader.get_test_df().head(5))


