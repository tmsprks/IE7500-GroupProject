
import inspect
import logging

from abc import ABC, abstractmethod
from utils.sa_app_config import SAAppConfig
from utils.sa_model_params import SAModelParams
from utils.sa_model_inference import SAModelInference
from utils.sa_model_checkpoint_util import SAModelCheckpointUtil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Abstract base class for sentiment models
class SASentimentModel(ABC):

    def __init__(self,
                 sa_app_config: SAAppConfig,
                 sa_model_param:SAModelParams=None):
        """sa_model_param is the initial param.  """
        self.sa_app_config = sa_app_config
        self.sa_model_params = sa_model_param

    def get_model_params(self) -> SAModelParams:
        return self.sa_model_params
    
    @abstractmethod
    def register(self, sa_model_param:SAModelParams=None) -> str:
        pass

    @abstractmethod
    def preprocess(self, sa_model_param:SAModelParams=None) -> None:
        """Preprocess text data using parameters in SAModelParams."""
        pass
    
    @abstractmethod
    def fit(self, sa_model_param:SAModelParams=None) -> None:
        """Train the model using parameters in SAModelParams."""
        pass
    
    @abstractmethod
    def predict(self, sa_model_param:SAModelParams=None) -> None:
        """Predict sentiment using parameters in SAModelParams."""
        pass

    @abstractmethod
    def inference (self, text_to_make_prediction_on: str=None) -> SAModelInference:
        """Predict the sentiment of the parameter text_to_make_prediction_on."""
        pass

    @abstractmethod
    def evaluate(self, sa_model_param:SAModelParams=None) -> None:
        """Evaluate model performance using parameters in SAModelParams."""
        pass

    @abstractmethod
    def summary(self, sa_model_param:SAModelParams=None) -> None:
        """Model summary"""
        pass

    ###
    ### Load the model given the file name
    ###
    @abstractmethod
    def load(self) -> None:
        """Load the model"""
        pass

    ###
    ### Load the model given the file name
    ###
    @abstractmethod
    def save(self) -> None:
        """Save the model"""
        pass

    ###
    ### Return the app config that was passed to the constructor
    ###

    def get_app_config(self) -> SAAppConfig:
        return self.sa_app_config
    
    def get_model_module_name(self) -> str:
        return(inspect.getmodule(inspect.currentframe()).__name__)

    def get_model_class_name(self) -> str:
        return(self.__class__.__name__)

    def get_checkpoint_file_name(self,
                                 model_module_name: str,
                                 model_class_name: str,
                                 file_name_suffix: str=None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {self.get_app_config()}")    

        clean_module_name = model_module_name.removeprefix("model.")
        sa_model_checkpoint_util = SAModelCheckpointUtil(self.get_app_config())
        model_checkpoint_path = sa_model_checkpoint_util.get_checkpoint_complete_path(clean_module_name, model_class_name, file_name_suffix)

        logger.info(f"{class_name}.{method_name}(): Checkpoint file name: {model_checkpoint_path}")  

        return model_checkpoint_path
    

    ###
    ### Run the model pipeline
    ###
    def run(self, sa_model_param:SAModelParams=None) -> None:
        self.register(sa_model_param)
        self.preprocess(sa_model_param)
        self.fit(sa_model_param)
        self.summary(sa_model_param)
        self.predict(sa_model_param)
        self.evaluate(sa_model_param)   
        self.save()    

