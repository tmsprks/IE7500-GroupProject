
from typing import List, Dict
import pandas as pd
import logging
from sa_model_config import SAModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelConfigLoader:
    def __init__(self, 
                 path_to_model_config: str = None):
        self.path_to_model_config = path_to_model_config
        self.list_of_model_config = None

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

    
    def load_model_config(self) -> List[SAModelConfig]:

        ### Use the path to model config passed into the constructor or assume the default model config file name
        ### and current working directory location
        model_config_file = self.path_to_model_config or SAModelConfig.get_default_model_config_file_name()

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

            self.list_of_model_config.append(SAModelConfig(model_name, module_name, class_name, train_csv, test_csv, validation_csv, self._process_model_params(model_params)))
        
        return self.list_of_model_config

    def get_list_of_model_config(self) -> List[SAModelConfig]:
        return self.list_of_model_config
    
    def find_model_config(self, 
                          model_module_name:str=None, 
                          model_class_name:str=None) -> SAModelConfig:
        list_of_model_config_len = len(self.list_of_model_config) or 0 
        found = list_of_model_config_len == 0
        i = 0
        return_value = None

        while not found:
            model_config = self.list_of_model_config[i]
            found = model_config.get_model_module_name() == model_module_name and model_config.get_model_class_name () == model_class_name
            if found:
                return_value = model_config
            i += 1
        
        return return_value