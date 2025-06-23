
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
        model_config_file = self.path_to_model_config or SAModelConfig.DEFAULT_MODEL_CONFIG_FILE_NAME

        ### Read the model config csv file
        try:
            model_config_df = pd.read_csv(model_config_file, 
                                          header=None, 
                                          names=[SAModelConfig.MODEL_CONFIG_MODEL_NAME, 
                                                 SAModelConfig.MODEL_CONFIG_MODULE_NAME, 
                                                 SAModelConfig.MODEL_CONFIG_CLASS_NAME,
                                                 SAModelConfig.MODEL_CONFIG_TRAIN_CSV,
                                                 SAModelConfig.MODEL_CONFIG_TEST_CSV,
                                                 SAModelConfig.MODEL_CONFIG_VALIDATION_CSV,
                                                 SAModelConfig.MODEL_CONFIG_MODEL_PARAMS])
        except FileNotFoundError:
            print(f"Error: File '{model_config_file}' not found.")
            return []
        
        ### Reinitializes the list_of_model_config
        self.list_of_model_config = []

        # Process each row in model config file
        for _, row in model_config_df.iterrows():
            model_name = row[SAModelConfig.MODEL_CONFIG_MODEL_NAME]
            module_name = row[SAModelConfig.MODEL_CONFIG_MODULE_NAME]
            class_name = row[SAModelConfig.MODEL_CONFIG_CLASS_NAME]
            train_csv = row[SAModelConfig.MODEL_CONFIG_TRAIN_CSV]
            test_csv = row[SAModelConfig.MODEL_CONFIG_TEST_CSV]
            validation_csv = row[SAModelConfig.MODEL_CONFIG_VALIDATION_CSV]
            model_params = row[SAModelConfig.MODEL_CONFIG_MODEL_PARAMS]

            self.list_of_model_config.append(SAModelConfig(model_name, module_name, class_name, train_csv, test_csv, validation_csv, self._process_model_params(model_params)))
        
        return self.list_of_model_config

    def get_list_of_model_config(self) -> List[SAModelConfig]:
        return self.list_of_model_config