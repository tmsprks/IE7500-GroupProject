
import importlib
import logging
import inspect

from utils.sa_model_params import SAModelParams
from utils.sa_data_loader import SADataLoader
from utils.kaggle_dataset import KaggleDataSet
from utils.sa_model_config_loader import SAModelConfigLoader
from utils.sa_app_config import SAAppConfig
from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_params_factory import SAModelParamsFactory
from utils.sa_model_inference import SAModelInference
from model.SASentimentModel import SASentimentModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelPipeline:
    def __init__(self, 
                 sa_app_config: SAAppConfig,
                 path_to_model_config_file:str=None):
        self.sa_app_config = sa_app_config
        self.path_to_model_config_file = path_to_model_config_file
        self.sa_model_config_loader = SAModelConfigLoader(self.sa_app_config)
        self.list_of_model_config = self.sa_model_config_loader.get_list_of_model_config()
        self.model_directory = self.sa_app_config.framework.model_dir
        self.map_of_instantiated_models = {}

        self._load_model_params()
        self._instantiate_models()
    
    def _model_lookup_key(self, model_module_name:str=None, model_class_name:str=None):
        return model_module_name + ":" + model_class_name
    
    def _load_model_params(self):
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name

        self.map_of_model_params = {}
        model_config_len = len(self.list_of_model_config)

        for i in range(model_config_len):
            model_config = self.list_of_model_config[i]
            
            train_file = model_config.get_model_train_csv_file_name()
            test_file = model_config.get_model_test_csv_file_name()
            validation_file = model_config.get_model_validation_csv_file_name()

            logger.info(f"{class_name}.{method_name}(): Loading model params for {model_config.get_model_module_name()}.{model_config.get_model_class_name()}")
            logger.info(f"{class_name}.{method_name}(): Model params specified train csv file: {train_file}, test csv file: {test_file}, validation csv file: {validation_file}")
            sa_model_params = SAModelParamsFactory.create_model_params(model_config.get_model_module_name(), model_config.get_model_class_name())
            self.map_of_model_params[self._model_lookup_key(model_config.get_model_module_name(), model_config.get_model_class_name())] = sa_model_params

    def _instantiate_models(self):
        model_config_len = len(self.list_of_model_config)

        method_class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{'---'*25}\n{method_class_name}.{method_name}(): Pre-instantiating {model_config_len} models defined in the model config")


        for i in range(model_config_len):
            model_config = self.list_of_model_config[i]
            module_name = model_config.get_model_module_name()
            class_name = model_config.get_model_module_name()

            module_path = f"{self.model_directory}.{module_name}"
            class_path = f"{self.model_directory}.{class_name}"
            logger.info(f"{method_class_name}.{method_name}(): Pre-instantiating model: {class_name} from module: {module_name}, model params: {model_config}")

            try:
                module = importlib.import_module(f"{module_path}")
                logger.info(f"Module obtained: {module_path}")

                model_class = getattr(module, class_name)
                logger.info(f"Class obtained: {class_name}")

                sa_model_params = self.map_of_model_params.get(self._model_lookup_key(module_name,class_name))
                logger.info(f"SA Model params obtained: {sa_model_params}")
                
                sa_sentiment_model = model_class(self.sa_app_config, sa_model_params)
                logger.info(f"Model obtained: {sa_sentiment_model}")
                
                self.map_of_instantiated_models[self._model_lookup_key(module_name, class_name)] = sa_sentiment_model
                logger.info(f"Model loaded: {module_name} : {class_name}: {sa_model_params}")
            except Exception as e:
                logger.error(f"Failed to load model {module_name} : {class_name}: {str(e)}")
            
            logger.info("="*25)

    def run_model_pipeline(self):
        logger.info(f"Running all model pipeline:  total model defined: {len(self.map_of_instantiated_models)}")

        for key, model in self.map_of_instantiated_models.items():
            sa_model_params = self.map_of_model_params.get(key)
            logger.info(f"{'===='*25}\nRunning model {key}")
            self._run_single_model_pipeline_with_model_params(model, sa_model_params)
            logger.info(f"Model {key} completed. \n{'===='*25}\n")


    def run_single_model_pipeline(self, 
                                  model_module_name:str=None,
                                  model_class_name:str=None):
        
            sa_sentiment_model = self.map_of_instantiated_models.get(self._model_lookup_key(model_module_name, model_class_name))
            sa_model_params = self.map_of_model_params.get(self._model_lookup_key(model_module_name, model_class_name))
            logger.info(f"{'===='*25}\nRunning single model pipeline: {sa_sentiment_model.__class__.__name__}")
            self._run_single_model_pipeline_with_model_params(sa_sentiment_model, sa_model_params)
            logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}\n{'===='*25}\n")

    
    def _run_single_model_pipeline_with_model_params (self, 
                                                      sa_sentiment_model: SASentimentModel=None,
                                                      sa_model_params: SAModelParams=None):
        try:

            ### Call the model's run method which will run the model pipeline
            sa_sentiment_model.run(sa_model_params)

        except Exception as e:
            logger.error(f"Error running model: {sa_sentiment_model.__class__.__name__}: {str(e)}")

    def load_single_model (self, 
                           model_module_name:str,
                           model_class_name:str) -> SASentimentModel:

        sa_sentiment_model = self.map_of_instantiated_models.get(self._model_lookup_key(model_module_name, model_class_name))
        sa_sentiment_model.load()
        return sa_sentiment_model

    def run_single_model_inference(self, 
                                   model_module_name:str,
                                   model_class_name:str,
                                   text_to_make_prediction_on: str) -> SAModelInference:

        sa_sentiment_model = self.map_of_instantiated_models.get(self._model_lookup_key(model_module_name, model_class_name))
        return sa_sentiment_model.inference(text_to_make_prediction_on)

