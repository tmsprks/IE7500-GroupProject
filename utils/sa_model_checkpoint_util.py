
import os
import sys
import logging
import inspect
from pathlib import Path
from utils.sa_app_config import SAAppConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelCheckpointUtil:    
    def __init__(self, 
                 sa_app_config: SAAppConfig):
        self.sa_app_config = sa_app_config

    def get_checkpoint_complete_path (self,
                                      model_module_name: str,
                                      model_class_name: str,
                                      file_name_extension: str=None,
                                      checkpoint_file_name_suffix: str=None) -> str:

        class_name = self.__class__.__name__
        self.file_name_extension = file_name_extension

        ###
        ### Don't pass in a file extension unless it was specified
        ###
        if self.file_name_extension is None:
            self.file_name_extension = ""

        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): app config: {self.sa_app_config}")

        ###
        ### From the app config, determine the model's checkpoint directory relative to app's current working directory
        ###
        app_config_model_checkpoint_directory = self.sa_app_config.model.checkpoint_dir
        
        current_working_dir = os.getcwd()
        
        model_checkpoint_directory = os.path.join(current_working_dir, app_config_model_checkpoint_directory)
        logger.info(f"{class_name}.{method_name}(): model checkpoint directory is {model_checkpoint_directory}")

        suffix = checkpoint_file_name_suffix is not None or ""
        model_checkpoint_file_name = model_module_name + "-" + model_class_name + suffix + self.file_name_extension
        model_checkpoint_complete_path = os.path.join(model_checkpoint_directory, model_checkpoint_file_name)
        logger.info(f"{class_name}.{method_name}(): model checkpoint complete path: {model_checkpoint_complete_path}")
        return model_checkpoint_complete_path

    def get_checkpoint_directory(self, model_module_name: str, model_class_name: str) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}():")

        model_checkpoint_name = f"{model_module_name}-{model_class_name}"
        # Construct full absolute path using pathlib
        checkpoint_root_dir = Path(self.sa_app_config.model.checkpoint_dir) / model_checkpoint_name
        checkpoint_root_dir_as_posix = checkpoint_root_dir.as_posix()

        logger.info(f"{class_name}.{method_name}(): model checkpoint path: {checkpoint_root_dir_as_posix}")

        return checkpoint_root_dir_as_posix