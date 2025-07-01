
import os
import csv
from typing import List, Dict, Tuple
import pandas as pd
import logging
from utils.sa_app_config import SAAppConfig
from utils.sa_app_config import TestConfig
from utils.sa_app_config import UtilsConfig
from utils.sa_app_config import ModelConfig
from utils.sa_app_config import FrameworkConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAAppConfigLoader:
    def __init__(self, 
                 path_to_model_config: str = "config/app_config.csv"):
        self.path_to_model_config = path_to_model_config
        self.flat_config = self._load_flat_config()
        self.sa_app_config = self._build_app_config(self.flat_config)
    
    def _load_flat_config(self) -> Dict[Tuple[str, str], str]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.flat_config = {}

        with open(self.path_to_model_config, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                component = row["component"].strip()
                key = row["key"].strip()
                value = row["value"].strip()

                if key.endswith("_dir") or key.startswith("rel_path"):
                    value = os.path.normpath(os.path.join(base_dir, value))

                self.flat_config[(component, key)] = value

        return self.flat_config

    def _build_app_config(self, flat_config: Dict[Tuple[str, str], str]) -> SAAppConfig:
        return SAAppConfig(
            utils=UtilsConfig(
                data_dir=flat_config[("utils", "path_to_data")],
                config_dir=flat_config[("utils", "path_to_config")]
            ),
            
            model=ModelConfig(
                checkpoint_dir=flat_config[("model", "path_to_checkpoint")]
            ),
            
            framework=FrameworkConfig(
                model_dir=flat_config[("framework", "path_to_model")]
            ),
            
            test=TestConfig(
                config_dir=flat_config[("test", "path_to_config")],
                output_dir=flat_config[("test","path_to_output_directory")]

            )
        )
        
    def get_app_config (self) -> SAAppConfig:
        return self.sa_app_config