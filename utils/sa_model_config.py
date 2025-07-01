
from typing import Any, Dict, List

class SAModelConfig:

    _DEFAULT_MODEL_CONFIG_FILE_NAME = "model_config.csv"
    _MODEL_CONFIG_MODEL_NAME = "Model Name"
    _MODEL_CONFIG_MODULE_NAME = "Model Module Name"
    _MODEL_CONFIG_CLASS_NAME = "Model Class Name"
    _MODEL_CONFIG_TRAIN_CSV = "Train CSV"
    _MODEL_CONFIG_TEST_CSV = "Test CSV"
    _MODEL_CONFIG_VALIDATION_CSV = "Validation CSV"
    _MODEL_CONFIG_MODEL_PARAMS = "Model Params"

    ###
    ### Order of the columns in the model_config.csv file where the
    ### 1st column is the model name
    ### 2nd column is the model module name
    ### 3rd column is the model class name
    ### .....
    ### last column is the model params

    _MODEL_CONFIG_COLUMN_NAMES = [_MODEL_CONFIG_MODEL_NAME, 
                                  _MODEL_CONFIG_MODULE_NAME, 
                                  _MODEL_CONFIG_CLASS_NAME, 
                                  _MODEL_CONFIG_TRAIN_CSV, 
                                  _MODEL_CONFIG_TEST_CSV, 
                                  _MODEL_CONFIG_VALIDATION_CSV, 
                                  _MODEL_CONFIG_MODEL_PARAMS]

    @staticmethod
    def get_default_model_config_file_name():
        return SAModelConfig._DEFAULT_MODEL_CONFIG_FILE_NAME 
    
    @staticmethod
    def get_model_config_column_names():
        return SAModelConfig._MODEL_CONFIG_COLUMN_NAMES.copy()  # Return a shallow copy    

    @staticmethod
    def get_model_config_model_name():
        return SAModelConfig._MODEL_CONFIG_MODEL_NAME 
    
    @staticmethod
    def get_model_config_model_module_name():
        return SAModelConfig._MODEL_CONFIG_MODULE_NAME 

    @staticmethod
    def get_model_config_model_class_name():
        return SAModelConfig._MODEL_CONFIG_CLASS_NAME 

    @staticmethod
    def get_model_config_model_train_csv_name():
        return SAModelConfig._MODEL_CONFIG_TRAIN_CSV 

    @staticmethod
    def get_model_config_model_test_csv_name():
        return SAModelConfig._MODEL_CONFIG_TEST_CSV 

    @staticmethod
    def get_model_config_model_validation_csv_name():
        return SAModelConfig._MODEL_CONFIG_VALIDATION_CSV 

    @staticmethod
    def get_model_config_model_params():
        return SAModelConfig._MODEL_CONFIG_MODEL_PARAMS 

    def __init__(self, 
                 model_name: str=None, 
                 model_module_name: str=None,
                 model_class_name: str=None,
                 model_train_csv_file_name: str=None,
                 model_test_csv_file_name: str=None,
                 model_validation_csv_file_name: str=None,
                 model_params: Dict[str, Any]=None):
        self.model_name = model_name
        self.model_module_name = model_module_name
        self.model_class_name = model_class_name
        self.model_train_csv_file_name = model_train_csv_file_name
        self.model_test_csv_file_name = model_test_csv_file_name
        self.model_validation_csv_file_name = model_validation_csv_file_name
        self.model_params = model_params

    def __str__(self):
        return f"{self.get_model_name()}/{self.get_model_module_name()}/{self.get_model_class_name()}/{self.get_model_train_csv_file_name()}/{self.get_model_test_csv_file_name()}/{self.get_model_validation_csv_file_name()}/{self.get_model_params()}"
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_model_module_name(self) -> str:
        return self.model_module_name
    
    def get_model_class_name(self) -> str:
        return self.model_class_name
    
    def get_model_train_csv_file_name(self) -> str:
        return self.model_train_csv_file_name
    
    def get_model_test_csv_file_name(self) -> str:
        return self.model_test_csv_file_name
    
    def get_model_validation_csv_file_name(self) -> str:
        return self.model_validation_csv_file_name
    
    def get_model_params(self) -> Dict[str, Any]:
        return self.model_params

    def verify_model_params(self, model_params_to_check:List[str]) -> bool:    
        missing = [key for key in model_params_to_check if key not in self.model_params]
        if missing:
            raise ValueError(f"Keys not found in model params: {missing}")
        return True
