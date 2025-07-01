import os
import sys
import inspect
import logging

from utils.sa_app_config import SAAppConfig
from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_config import SAModelConfig
from utils.sa_model_config_loader import SAModelConfigLoader
from utils.sa_data_loader import SADataLoader
from utils.sa_model_params import SAModelParams
from utils.kaggle_dataset import KaggleDataSet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelParamsFactory:

    def __init__(self):
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        raise NotImplementedError(f"{class_name}.{method_name}: This is a utility class and cannot be instantiated.")

    @staticmethod
    def create_model_params_using_config(sa_app_config: SAAppConfig,
                                         model_config: SAModelConfig,
                                         model_module_name: str,
                                         model_class_name: str) -> SAModelParams:
        
        ###
        ### Get the train, test, and validation CSV file name associated with this model
        ###
        train_file = model_config.get_model_train_csv_file_name()
        test_file = model_config.get_model_test_csv_file_name()
        validation_file = model_config.get_model_validation_csv_file_name()

        ### Load the train, test and validation data file specific to the model
        sa_data_loader = SADataLoader(sa_app_config, train_file, test_file, validation_file)
        sa_data_loader.load_data(KaggleDataSet.get_kaggle_column_names())

        ### Construct the model parameter object
        sa_model_params = SAModelParams(sa_data_loader, model_config)

        logger.info(f"Model params created: {sa_model_params}")

        return sa_model_params


    @staticmethod
    def create_model_params(model_module_name: str,
                            model_class_name: str) -> SAModelParams:
        
        logger.info(f"SAModelParamsFactory.create_model_params({model_module_name}, {model_class_name}")
        sa_app_config_loader = SAAppConfigLoader()
        sa_app_config = sa_app_config_loader.get_app_config()
        sa_model_config_loader = SAModelConfigLoader(sa_app_config)

        return SAModelParamsFactory.create_model_params_using_config(sa_app_config,
                                                                     sa_model_config_loader.find_model_config(model_module_name, model_class_name),
                                                                     model_module_name, 
                                                                     model_class_name)