import os
import sys
import logging

# Add project root to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_params_factory import SAModelParamsFactory
from model.SALSTMModel import SALSTMModel  # ✅ Use the LSTM model

print("Imported all modules successfully")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    ### ✅ Specify the module and class name for the LSTM model
    model_module_name = "SALSTMModel"
    model_class_name = "SALSTMModel"

    # Create model parameters
    sa_model_params = SAModelParamsFactory.create_model_params(model_module_name, model_class_name)

    logger.info(f"Start running model: {model_module_name}:{model_class_name}") 

    # ✅ Instantiate the LSTM-based sentiment model
    sa_sentiment_model = SALSTMModel(SAAppConfigLoader().get_app_config(), sa_model_params)

    # ✅ Run the pipeline: register, preprocess, fit, evaluate, save, etc.
    sa_sentiment_model.run(sa_model_params)

    logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}") 

if __name__ == "__main__":
    main()