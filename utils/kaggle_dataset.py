
import os
from typing import List, Any
import pandas as pd
from  .kaggle_data_loader import KaggleDataLoader

class KaggleDataSet:

    ### KAGGLE_POLARITY_COLUMN_NAME is the label from the Kaggle dataset: {1, 2}
    _KAGGLE_POLARITY_COLUMN_NAME = "Kaggle_Label"

    ### POLARITY_COLUMN_NAME is the transformed label from {1, 2} to {0, 1} which is 
    ### appropriate for binary classification
    _POLARITY_COLUMN_NAME = "Label"
    
    _TITLE_COLUMN_NAME = "Title"
    _REVIEW_COLUMN_NAME = "Review"

    _POLARITY_VALUE_1 = 1
    _POLARITY_VALUE_2 = 2

    KAGGLE_COLUMN_NAMES = [_KAGGLE_POLARITY_COLUMN_NAME, 
                           _TITLE_COLUMN_NAME,
                           _REVIEW_COLUMN_NAME]

    @staticmethod
    def get_kaggle_polarity_value_1 () -> int:
        return KaggleDataSet._POLARITY_VALUE_1
    
    @staticmethod
    def get_kaggle_polarity_value_2 () -> int:
        return KaggleDataSet._POLARITY_VALUE_2

    @staticmethod
    def get_kaggle_column_names() -> List[str]:
        return KaggleDataSet.KAGGLE_COLUMN_NAMES.copy()  # Return a shallow copy    

    @staticmethod
    def get_title_column_name() -> str:
        return KaggleDataSet._TITLE_COLUMN_NAME
        
    @staticmethod
    def get_kaggle_polarity_column_name() -> str:
        return KaggleDataSet._KAGGLE_POLARITY_COLUMN_NAME      
    
    @staticmethod
    def get_polarity_column_name() -> str:
        return KaggleDataSet._POLARITY_COLUMN_NAME

    @staticmethod
    def get_review_column_name() -> str:
        return KaggleDataSet._REVIEW_COLUMN_NAME
        
    def __init__(self, kaggle_data_loader: KaggleDataLoader):
        self.kaggle_data_loader = kaggle_data_loader
        self.train_df = kaggle_data_loader.get_train_df()
        self.test_df = kaggle_data_loader.get_test_df()

    def get_train_df(self) -> pd.DataFrame:
        return self.train_df
    
    def get_test_df(self) -> pd.DataFrame:
        return self.test_df
    
    def get_train_value_count(self) -> Any:
        return self.get_train_df()[KaggleDataSet.get_kaggle_polarity_column_name()].value_counts()

    def get_test_value_count(self) -> Any:
        return self.get_test_df()[KaggleDataSet.get_kaggle_polarity_column_name()].value_counts()
    
# Example usage:
if __name__ == "__main__":
    print("KaggleDataSet")