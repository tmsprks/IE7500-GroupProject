import os
import sys
import logging

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_params_factory import SAModelParamsFactory
from model.SALSTMModel import SALSTMModel  # ✅ Use LSTM model

print("Imported all modules successfully")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    model_module_name = "SALSTMModel"
    model_class_name = "SALSTMModel"

    sa_model_params = SAModelParamsFactory.create_model_params(model_module_name, model_class_name)
    logger.info(f"Start running model: {model_module_name}:{model_class_name}") 

    sa_sentiment_model = SALSTMModel(SAAppConfigLoader().get_app_config(), sa_model_params)

    # ✅ Load pretrained model
    sa_sentiment_model.load()

    # Example reviews to test inference
    new_reviews = [
        "I don't love this product. It is not a bargain too!",
        "Man, this product is not terrible, pretty good quality!",
        "I would not buy another one even though the product is not bad",
        "Not sure if I like it. I may grow to like it",
        "Absolutely amazing",
        "Absolutely terrible",
        "Absolutely insane",
        "Absolutely worth it",
        "Absolutely not worth it",
        "Did it’s purpose but the plug will randomly pop out so then you have to hurry up and close it and then air back up so beaware",
        "I'm a big heavy(fat) guy... definitely worth the price and is comfortable. I'd buy it again regardless.",
        "I bought this via same day delivery... glad I bought it.",
        "I opened the package... a few hours later it was flat.",
        "If you can get it aired up enough... very poor design. Would not buy again.",
        "As a camping individual, this bed is a need to have. Worth the cost, and love the size"
    ]

    predictions = []
    for review in new_reviews:
        predictions.append(sa_sentiment_model.inference(review))

    logger.info("\n\nModel predictions:\n")
    for prediction in predictions:
        logger.info(f"{prediction}\n")

    logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}") 


if __name__ == "__main__":
    main()