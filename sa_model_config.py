
from typing import Dict

class SAModelConfig:

    DEFAULT_MODEL_CONFIG_FILE_NAME = "model_config.csv"
    MODEL_CONFIG_MODEL_NAME = "Model Name"
    MODEL_CONFIG_MODULE_NAME = "Model Module Name"
    MODEL_CONFIG_CLASS_NAME = "Model Class Name"
    MODEL_CONFIG_TRAIN_CSV = "Train CSV"
    MODEL_CONFIG_TEST_CSV = "Test CSV"
    MODEL_CONFIG_VALIDATION_CSV = "Validation CSV"
    MODEL_CONFIG_MODEL_PARAMS = "Model Params"

    def __init__(self, 
                 model_name: str=None, 
                 model_module_name: str=None,
                 model_class_name: str=None,
                 model_train_csv_file_name: str=None,
                 model_test_csv_file_name: str=None,
                 model_validation_csv_file_name: str=None,
                 model_params: Dict=None):
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
    
    def get_model_params(self) -> Dict:
        return self.model_params