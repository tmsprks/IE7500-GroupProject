
from abc import ABC, abstractmethod
from sa_model_params import SAModelParams

# Abstract base class for sentiment models
class SASentimentModel(ABC):

    def __init__(self, sa_model_param:SAModelParams=None):
        self.params = sa_model_param

    def get_model_params(self) -> SAModelParams:
        return self.params
    
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
    def evaluate(self, sa_model_param:SAModelParams=None) -> None:
        """Evaluate model performance using parameters in SAModelParams."""
        pass

