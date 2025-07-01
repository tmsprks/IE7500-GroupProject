import os
import sys
import numpy as np
import logging
import inspect
import re

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)
print("Project directory added:", project_dir)

# Debug utils import
try:
    import utils
    print("utils package found at:", utils.__file__)
except ImportError as e:
    print("Import error for utils:", e)

from model.SASentimentModel import SASentimentModel
from utils.kaggle_dataset import KaggleDataSet

from framework.sa_model_pipeline import SAModelPipeline
from utils.sa_model_config_loader import SAModelConfigLoader
from utils.sa_data_loader import SADataLoader
from utils.sa_model_params import SAModelParams
from model.SASelfAttentionModel import SASelfAttentionModel
from model.SARnnModel import SARnnModel
from utils.sa_model_params_factory import SAModelParamsFactory
from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_app_config import SAAppConfig

print("Imported all utils modules successfully")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


### Extract epoch number from checkpoint file
### example: 'epoch03' will return 3, else 0 if no number is found

def extract_initial_epoch_from_checkpoint(path):
    match = re.search(r"epoch(\d+)", path)
    return int(match.group(1)) if match else 0


def main():

    ### TOGGLE
    
    ### If no existing model to load, set LOAD_EXISTING_MODEL to False (train from scratch)
    ### If there is an existing model to load, set LOAD_EXISTING_MODEL to True and load the CHECKPOINT_PATH
    
    LOAD_EXISTING_MODEL = True
    CHECKPOINT_FILE = "model_checkpoint_epoch02_valacc0.5286.keras"  # adjust to the saved keras model

    ### Hard coding the checkpoint file name for testing only.
    ### Get the checkpoint directory from the app config.
    sa_app_config = SAAppConfigLoader().get_app_config()
    checkpoint_path = os.path.join(sa_app_config.model.checkpoint_dir, CHECKPOINT_FILE)
    
    ### 
    ### To run your model, 
    ###
    ### 1) Change model_module_name to match your model's module name, i.e., the file name
    ### 2) Change model_class_name to match  your model's class name like SASelfAttentionModel
    ### 3) Look for this line at the bottom of the method,
    ###    sa_sentiment_model = SASelfAttentionModel(sa_model_params)
    ###    Change SASelfAttentionModel to match your model's class.
    ###    That's it.
    ### 4) Then run this notebook from start to finish.
    ###


    ###
    ### Modify model_module_name and model_class_name to match your model's 
    ### module and class name and then change the model instance to match your model class below
    ###
    model_module_name = "SARnnModel"
    model_class_name = "SARnnModel"

    logger.info(f"Start running model: {model_module_name}:{model_class_name}") 

    ### Construct the model parameter object
    sa_model_params = SAModelParamsFactory.create_model_params(model_module_name, model_class_name)

    ###
    ### CHANGE THE MODEL TO YOUR MODEL CLASS!!
    ### 

    sa_sentiment_model = SARnnModel(sa_app_config, sa_model_params)    
    ###
    ### CHANGE THE MODEL TO YOUR MODEL CLASS!!
    ### 

    logger.info(f"Start running model: {model_module_name}:{model_class_name}") 

    ### Checkpoint
    if LOAD_EXISTING_MODEL:    
        ### Register model/preprocessing
        sa_sentiment_model.register(sa_model_params)
        sa_sentiment_model.preprocess(sa_model_params)
        initial_epoch = extract_initial_epoch_from_checkpoint(checkpoint_path) ### epoch number from checkpoint filename
        sa_sentiment_model.load_model_from_file(checkpoint_path) ### load saved model weights
        logger.info(f"Resuming training from epoch {initial_epoch}")
        sa_sentiment_model.fit(sa_model_params, resume_training=True, initial_epoch=initial_epoch) ### continue training
        sa_sentiment_model.predict(sa_model_params)
        sa_sentiment_model.evaluate(sa_model_params)
    ### Call the SASentimentModel's run() which will run the model pipeline
    else:
        sa_sentiment_model.run(sa_model_params)
    ###
    ###
    ###


    logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}") 


if __name__ == "__main__":
    main()
