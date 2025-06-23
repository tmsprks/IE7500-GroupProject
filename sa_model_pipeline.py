
import os
import importlib
import logging

from sa_model_config import SAModelConfig
from sa_model_params import SAModelParams
from sa_data_loader import SADataLoader
from kaggle_dataset import KaggleDataSet
from sa_model_config_loader import SAModelConfigLoader
from SASentimentModel import SASentimentModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelPipeline:
    def __init__(self, path_to_model_config_file:str=None):
        self.path_to_model_config_file = path_to_model_config_file
        self.sa_model_config_loader = SAModelConfigLoader()
        self.list_of_model_config = self.sa_model_config_loader.load_model_config()
        self.map_of_instantiated_models = {}

        self._load_model_params()
        self._instantiate_models()
    
    def _model_lookup_key(self, model_module_name:str=None, model_class_name:str=None):
        return model_module_name + ":" + model_class_name
    
    def _load_model_params(self):
        self.map_of_model_params = {}
        model_config_len = len(self.list_of_model_config)
        column_names = [KaggleDataSet.POLARITY_COLUMN_NAME, 
                        KaggleDataSet.TITLE_COLUMN_NAME,
                        KaggleDataSet.REVIEW_COLUMN_NAME]

        cwd = os.getcwd()
        for i in range(model_config_len):
            model_config = self.list_of_model_config[i]
            
            train_file = model_config.get_model_train_csv_file_name()
            test_file = model_config.get_model_test_csv_file_name()
            validation_file = model_config.get_model_validation_csv_file_name()

            path_to_train_csv_file = os.path.join(cwd, train_file)
            path_to_test_csv_file = os.path.join(cwd, test_file)
            path_to_validation_csv_file = os.path.join(cwd, validation_file)

            sa_data_loader = SADataLoader(path_to_train_csv_file, path_to_test_csv_file, path_to_validation_csv_file)
            sa_data_loader.load_data(column_names)

            sa_model_params = SAModelParams(sa_data_loader, model_config)
            self.map_of_model_params[self._model_lookup_key(model_config.get_model_module_name() ,model_config.get_model_class_name())] = sa_model_params

    def _instantiate_models(self):
        model_config_len = len(self.list_of_model_config)

        for i in range(model_config_len):
            model_config = self.list_of_model_config[i]
            module_name = model_config.get_model_module_name()
            class_name = model_config.get_model_module_name()

            logger.info(f"Instantiating: {class_name} from module: {module_name}, model params: {model_config}")

            try:
                module = importlib.import_module(f"{module_name}")
                logger.info(f"Module obtained: {module}")

                model_class = getattr(module, class_name)
                logger.info(f"Class obtained: {model_class}")

                sa_model_params = self.map_of_model_params.get(self._model_lookup_key(module_name,class_name))
                logger.info(f"SA Model params obtained: {sa_model_params}")
                
                model = model_class(sa_model_params)
                logger.info(f"Model obtained: {model}")
                
                self.map_of_instantiated_models[self._model_lookup_key(module_name, class_name)] = model
                logger.info(f"Loaded model: {module_name} : {class_name}: {sa_model_params}")
            except Exception as e:
                logger.error(f"Failed to load model {class_name}: {str(e)}")
            
            logger.info("="*100)

    

    def run_model_pipeline(self):
        for key, model in self.map_of_instantiated_models.items():
            logger.info(f"Running model: {model.__class__.__name__}")
            sa_model_params = self.map_of_model_params.get(key)
            self._run_single_model_pipeline_with_model_params(model, sa_model_params)


    def run_single_model_pipeline(self, 
                                  model_module_name:str=None,
                                  model_class_name:str=None):
        
            model = self.map_of_instantiated_models.get(self._model_lookup_key(model_module_name, model_class_name))
            logger.info(f"Running model: {model.__class__.__name__}")
            sa_model_params = self.map_of_model_params.get(self._model_lookup_key(model_module_name, model_class_name))
            self._run_single_model_pipeline_with_model_params(model, sa_model_params)

    
    def _run_single_model_pipeline_with_model_params (self, 
                                                      sa_sentiment_model: SASentimentModel=None,
                                                      sa_model_params: SAModelParams=None):
        try:
            logger.info(sa_sentiment_model.register(sa_model_params))
            sa_sentiment_model.preprocess(sa_model_params)
            sa_sentiment_model.fit(sa_model_params)
            sa_sentiment_model.predict(sa_model_params)
            sa_sentiment_model.evaluate(sa_model_params)
            logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error running model: {sa_sentiment_model.__class__.__name__}: {str(e)}")
