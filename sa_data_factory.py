
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from kaggle_dataset import KaggleDataSet

class SADataFactory:
    DEFAULT_TRAIN_TEST_SPLIT_RATIO = 0.2
    
    def __init__(self, 
                 kaggle_dataset: KaggleDataSet, 
                 train_test_split_ratio:float = DEFAULT_TRAIN_TEST_SPLIT_RATIO):
        self.kaggle_dataset = kaggle_dataset
        self.train_test_ratio = train_test_split_ratio
        self.train_df = None
        self.validation_df = None

        ###
        ### Perform the explicit train/validation split since the Kaggle dataset doesn't
        ### have a validation csv file, only the train and test csv file
        ###
        self._train_validation_split(kaggle_dataset, train_test_split_ratio)

    def _train_validation_split(self, 
                                kaggle_dataset: KaggleDataSet, 
                                train_test_split_ratio:float):
        self.train_df, self.validation_df = train_test_split(kaggle_dataset.get_train_df(), 
                                                             test_size=train_test_split_ratio)

    def get_train_df(self) -> pd.DataFrame:
        return self.train_df
    
    def get_test_df(self) -> pd.DataFrame:
        return self.kaggle_dataset.get_test_df()
    
    def get_validation_df(self) -> pd.DataFrame:
        return self.validation_df


