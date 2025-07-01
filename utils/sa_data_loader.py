
import os
import logging
import inspect
import pandas as pd
import kagglehub
from typing import Callable, Union

from utils.binary_label_transformer import BinaryLabelTransformer
from utils.kaggle_dataset import KaggleDataSet
from utils.sa_app_config import SAAppConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SADataLoader:
    
    def __init__(self, 
                 sa_app_config: SAAppConfig,
                 train_csv_file: str = None, 
                 test_csv_file: str = None, 
                 validation_csv_file: str = None,
                 transform: Union[Callable[[int], int], BinaryLabelTransformer] = BinaryLabelTransformer()):
        self.sa_app_config = sa_app_config
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.validation_csv_file = validation_csv_file
        self.train_df = None
        self.test_df = None
        self.validation_df = None

        # Set transform: ensure it's callable
        if isinstance(transform, BinaryLabelTransformer):
            self.transform = transform.transform  # Use the default transformer
        elif callable(transform):
            self.transform = transform
        else:
            raise ValueError(
                "transform must be a callable function or BinaryLabelTransformer instance"
            )
        
        # Debug: Confirm transform is set correctly
        print(f"Transform set to: {self.transform}, Type: {type(self.transform)}")

        
    def __str__(self):
        return f"App config: {self.sa_app_config}, Train csv: {self.path_to_train_csv_file}, size: {len(self.get_train_df())}, Test csv: {self.path_to_test_csv_file}, size: {len(self.get_test_df())}, Test csv: {self.path_to_validation_csv_file}, size: {len(self.get_validation_df())}"

    def load_data(self, column_names: list[str]):
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name

        logger.info(f"{class_name}.{method_name}(): train csv file: {self.train_csv_file}, test csv file: {self.test_csv_file}, validation csv file: {self.validation_csv_file}")
      
        data_directory = self.sa_app_config.utils.data_dir
        if data_directory == None:
            data_directory = os.getcwd()
            logger.info(f"{class_name}.{method_name}(): data directory is NONE.  Using {data_directory}")

        path_to_train_csv_file = os.path.join(data_directory, self.train_csv_file)
        logger.info(f"{class_name}.{method_name}():  Directory: {data_directory}")
        logger.info(f"Path to train csv file: {path_to_train_csv_file}")
        self.train_df = pd.read_csv(path_to_train_csv_file, header=None, names=column_names)

        ### We need to map the default label values {1, 2} in the Kaggle dataset to {0: 1} for binary classification
        ### Store the transformed label in the KaggleDataSet.get_polarity_column_name() column
        self.train_df[KaggleDataSet.get_polarity_column_name()] = self.train_df[KaggleDataSet.get_kaggle_polarity_column_name()].apply(self.transform)

        path_to_test_csv_file = os.path.join(data_directory, self.test_csv_file)
        logger.info(f"Path to test csv file: {path_to_test_csv_file}")
        self.test_df = pd.read_csv(path_to_test_csv_file, header=None, names=column_names)

        ### We need to map the default label values {1, 2} in the Kaggle dataset to {0: 1} for binary classification
        ### Store the transformed label in the KaggleDataSet.get_polarity_column_name() column
        self.test_df[KaggleDataSet.get_polarity_column_name()] = self.test_df[KaggleDataSet.get_kaggle_polarity_column_name()].apply(self.transform)


        if self.validation_csv_file is not None:
            path_to_validation_csv_file = os.path.join(data_directory, self.validation_csv_file)
            logger.info(f"Path to validation csv file: {path_to_validation_csv_file}")
            self.validation_df = pd.read_csv(path_to_validation_csv_file, header=None, names=column_names)
            
            ### We need to map the default label values {1, 2} in the Kaggle dataset to {0: 1} for binary classification
            ### Store the transformed label in the KaggleDataSet.get_polarity_column_name() column
            self.validation_df[KaggleDataSet.get_polarity_column_name()] = self.validation_df[KaggleDataSet.get_kaggle_polarity_column_name()].apply(self.transform)

        print(f"Train size: {len(self.train_df)}")
        print(f"Test size: {len(self.test_df)}")

        if self.validation_df is not None:
            print(f"Validation size: {len(self.validation_df)}")
        else:
            print("Validation size: 0")

        return (self.train_csv_file, self.test_csv_file, self.validation_csv_file)
    
    def get_train_df(self):
        return self.train_df
    
    def get_test_df(self):
        return self.test_df
    
    def get_validation_df(self):
        return self.validation_df


