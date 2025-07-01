
from typing import List, Dict
import pandas as pd
import os
import logging
import inspect
from utils.sa_model_config import SAModelConfig
from utils.sa_app_config import SAAppConfig
from utils.sa_data_generator import SADataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelConfigLoader:
    def __init__(self, 
                 sa_app_config: SAAppConfig):
        self.sa_app_config = sa_app_config
        self.list_of_model_config = None

        self._load_model_config()

    def _process_model_params(self, model_params_as_string: str) -> Dict:

        """Parse parameter string (e.g., 'embedding=100,vocab_size=200') into a dictionary."""
        return_value = {}

        if not model_params_as_string or pd.isna(model_params_as_string):
            return return_value
        
        try:
            # Split by comma and then by equals sign
            for param in model_params_as_string.split(','):
                param = param.strip()
                if '=' not in param:
                    logger.warning(f"Invalid parameter format: {param}. Skipping.")
                    continue
                key, value = param.split('=', 1)
                key = key.strip()
                try:
                    # Attempt to convert value to int or float if possible
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string if not numeric
                    return_value[key] = value
                except Exception as e:
                    logger.warning(f"Error parsing parameter {param}: {str(e)}. Skipping.")
        except Exception as e:
            logger.error(f"Error parsing parameters '{model_params_as_string}': {str(e)}")
        
        return return_value

    
    def _load_model_config(self) -> List[SAModelConfig]:
        
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name

        logger.info(f"{class_name}.{method_name}(): app config: {self.sa_app_config}")

        model_config_directory = self.sa_app_config.utils.config_dir
        
        if model_config_directory == None:
            model_config_directory = os.getcwd()
            logger.info(f"{class_name}.{method_name}(): model config directory is {model_config_directory}")
        
        model_config_file = os.path.join(model_config_directory, SAModelConfig.get_default_model_config_file_name())
        logger.info(f"{class_name}.{method_name}(): model config file: {model_config_file}")

        ### Read the model config csv file
        try:
            model_config_df = pd.read_csv(model_config_file, 
                                          header=None, 
                                          names=SAModelConfig.get_model_config_column_names())
        except FileNotFoundError:
            print(f"Error: File '{model_config_file}' not found.")
            return []
        
        ### Reinitializes the list_of_model_config
        self.list_of_model_config = []

        # Process each row in model config file
        for _, row in model_config_df.iterrows():
            model_name = row[SAModelConfig.get_model_config_model_name()]
            module_name = row[SAModelConfig.get_model_config_model_module_name()]
            class_name = row[SAModelConfig.get_model_config_model_class_name()]
            train_csv = row[SAModelConfig.get_model_config_model_train_csv_name()]
            test_csv = row[SAModelConfig.get_model_config_model_test_csv_name()]
            validation_csv = row[SAModelConfig.get_model_config_model_validation_csv_name()]
            model_params = row[SAModelConfig.get_model_config_model_params()]

            sa_model_config = SAModelConfig(model_name, module_name, class_name, train_csv, test_csv, validation_csv, self._process_model_params(model_params))
            logger.info(f"{class_name}.{method_name}(): Model config read: {sa_model_config}")
            self.list_of_model_config.append(sa_model_config)
        
        logger.info(f"{class_name}.{method_name}(): Number of model config read: {len(self.list_of_model_config)}")

        return self.list_of_model_config

    def get_list_of_model_config(self) -> List[SAModelConfig]:
        return self.list_of_model_config
    
    def find_model_config(self, 
                          model_module_name:str=None, 
                          model_class_name:str=None) -> SAModelConfig:
    
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
    
        list_of_model_config_len = len(self.list_of_model_config) or 0 
        found = list_of_model_config_len == 0
        i = 0
        return_value = None

        logger.info(f"{class_name}.{method_name}(): Looking for [{model_module_name}.{model_class_name}], len of list of model config: {list_of_model_config_len}")

        while not found:
            model_config = self.list_of_model_config[i]
            found = model_config.get_model_module_name() == model_module_name and model_config.get_model_class_name () == model_class_name
            if found:
                return_value = model_config
            i += 1
        
        return return_value