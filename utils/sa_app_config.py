
from pydantic import BaseModel, DirectoryPath
from typing import Dict, Tuple

class TestConfig(BaseModel):
    config_dir: DirectoryPath
    output_dir: DirectoryPath

class UtilsConfig(BaseModel):
    data_dir: DirectoryPath
    config_dir: DirectoryPath

class ModelConfig(BaseModel):
    checkpoint_dir: DirectoryPath

class FrameworkConfig(BaseModel):
    model_dir: DirectoryPath

class SAAppConfig(BaseModel):
    test: TestConfig
    utils: UtilsConfig
    model: ModelConfig
    framework: FrameworkConfig

    def __str__(self):
        return f"Test config dir: {self.test.config_dir}, output dir: {self.test.output_dir}.  Utils data dir: {self.utils.data_dir}, config dir: {self.utils.config_dir}.   Model checkpoint: {self.model.checkpoint_dir}.  Framework model dir: {self.framework.model_dir}"