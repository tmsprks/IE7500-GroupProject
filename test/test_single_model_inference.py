import os
import sys
import logging

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)


from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_params_factory import SAModelParamsFactory
from model.SAxyzBERTSentimentModel import SAxyzBERTSentimentModel

print("Imported all modules successfully")


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
    ###model_module_name = "SASelfAttentionModel"
    ###model_class_name = "SASelfAttentionModel"
    model_module_name = "SAxyzBERTSentimentModel"
    model_class_name = "SAxyzBERTSentimentModel"
    
    ### Construct the model parameter object
    sa_model_params = SAModelParamsFactory.create_model_params(model_module_name, model_class_name)

    logger.info(f"Start running model: {model_module_name}:{model_class_name}") 

    ###
    ### CHANGE THE MODEL TO YOUR MODEL CLASS!!
    ### 

    sa_sentiment_model = SAxyzBERTSentimentModel(SAAppConfigLoader().get_app_config(), sa_model_params)
    
    ###
    ###
    ###

    ###
    ### Load the pre-trained model
    ###
    sa_sentiment_model.load()

    ###
    ### Here are some unseen data/reviews that we can use to test our models 
    ###
    new_reviews = ["I don't love this product. It is not a bargain too!", 
                "Man, this product is not terrible, pretty good quality!",
                "I would not buy another one eventhough the product is not bad",
                "Not sure if i like it.  I may grow to like it",
                "Absolutely amazing",
                "Absolutely terrible",
                "Absolutely insane",
                "Absolutely worth it",
                "Absolutely not worth it",
                ### Reviews below are from an acutal amazon product:   https://www.amazon.com/product-reviews/B0768LQLGH
                "Did it’s purpose but the plug will randomly pop out so then you have to hurry up and close it and then air back up so beaware",
                "I'm a big heavy(fat) guy that's why I got this mattress. It's a huge improvement over my other regular mattress. My issue with it is it needs to be refilled daily, sometimes 2-3 timess. It doesn't have a hole and I don't feel air leaking from the fill port. It's weird. Definitely worth the price and is comfortable. I'd buy it again regardless.",
                "I bought this via same day delivery because I found myself at my friends house and could not for the life of me get comfortable in her guest bed. I had committed to staying with her for a week, and had I not gotten any sleep during that time, we would have had a terrible visit. So this mattress to the rescue. I bought duct tape just in case the 15$ mattress I bought was leaking air - BUT IT DIDNT. This bed was comfortable enough to sleep on nightly for a week, and then small enough to fold up and put in the trunk of my car for next time. I am glad I bought it. As a sidenote however, it did not come with an airpump so you will need to get one (she had one already) but you could theorhetically blow it up like a raft if you dont need it too firm.",
                "I opened the package and carefully put the air mattress aside on a sofa to air out. I didn't get to using it for a month. When I did pump it up, a few hours later it was flat. No kids or pets to have messed with it. I should have tested it earlier but I've never had an air mattress deflate like that before. I've been using air mattresses for four years now with no problems. They generally last 18 months or so with daily use. That's pretty good for a camping air mattress! I'll replace this one from a brick and mortar store.",
                "If you can get it aired up enough it’s difficult to get the stopper in for it to keep air in. Very poor design. Would not buy again. Do not recommend this product.",
                "as a camping individual, this bed is a need to have. Worth the cost, and love the size"
                ]


    ###
    ### Then for all the new reviews, let's see what the model predicts as the sentiment of the reviews
    ### We use the model pipeline to invoke the model base on module name and class name
    ### Predictions will contain a list of SAModelInference
    ###
    predictions = []
    for i in range (len(new_reviews)):
        predictions.append(sa_sentiment_model.inference(new_reviews[i]))

    logger.info("\n\nModel predictions:\n")

    for j in range(len(predictions)):
        logger.info(f"{predictions[j]}\n")


    logger.info(f"Finished running model: {sa_sentiment_model.__class__.__name__}") 



if __name__ == "__main__":
    main()
