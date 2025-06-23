
from typing import Dict
import pandas as pd
from sa_data_loader import SADataLoader
from sa_model_config import SAModelConfig

class SAModelParams:
    def __init__(self,
                 sa_data_loader: SADataLoader,
                 sa_model_config: SAModelConfig):
        self.sa_data_loader = sa_data_loader
        self.sa_model_config = sa_model_config
        self.model_params = sa_model_config.get_model_params()
    
    def __str__(self):
        return f"{self.sa_model_config}"

    def get_model_param(self, key: str=None):
        return self.model_params.get(key)
    
    def update_model_param(self, key: str=None, value=None):
        self.model_params[key]=value

    def get_train_df(self) -> pd.DataFrame:
        return self.sa_data_loader.get_train_df()
    
    def get_test_df(self) -> pd.DataFrame:
        return self.sa_data_loader.get_test_df()
    
    def get_validation_df(self) -> pd.DataFrame:
        return self.sa_data_loader.get_test_df()
    
    def get_all_model_params(self) -> Dict:
        return self.model_params