import os
import sys
import logging

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
###print("Current working directory:", os.getcwd())
###print("sys.path:", sys.path)
###print("Project directory added:", project_dir)

from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_params_factory import SAModelParamsFactory
from model.SAxyzBERTSentimentModel import SAxyzBERTSentimentModel

print("Imported all utils modules successfully")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():

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
    ##model_module_name = "SASelfAttentionModel"
    ##model_class_name = "SASelfAttentionModel"
    model_module_name = "SAxyzBERTSentimentModel"
    model_class_name = "SAxyzBERTSentimentModel"
    ###model_module_name = "SADistilBERTSentimentModel"
    ###model_class_name = "SADistilBERTSentimentModel"

    ### Construct the model parameter object
    sa_model_params = SAModelParamsFactory.create_model_params(model_module_name, model_class_name)

    logger.info(f"Start running model: {model_module_name}:{model_class_name}") 

    ###
    ### CHANGE THE MODEL TO YOUR MODEL CLASS!!
    ### 

    sa_sentiment_model = SAxyzBERTSentimentModel (SAAppConfigLoader().get_app_config(), sa_model_params)
    
    ###
    ###
    ###

    ###
    ### Call the SASentimentModel's run() which will run the model pipeline
    ###
    sa_sentiment_model.run(sa_model_params)

    logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}") 



if __name__ == "__main__":
    main()
