
import os
import sys
import numpy as np
import logging
import inspect

### Simple light weight Python UI toolkit
import gradio as gr

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from framework.sa_model_pipeline import SAModelPipeline
from utils.sa_model_config_loader import SAModelConfigLoader
from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_model_inference import SAModelInference

print("Imported all modules successfully")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sa_app_config_loader = SAAppConfigLoader()
sa_model_pipeline = SAModelPipeline(sa_app_config_loader.get_app_config())
sa_model_config_loader = SAModelConfigLoader(sa_app_config_loader.get_app_config())

list_of_model_config = sa_model_config_loader.get_list_of_model_config()
list_of_models_for_selection = []
map_of_models_to_selection_list = {}
map_of_models = {}

for i in range(len(list_of_model_config)):
    model_config = list_of_model_config[i]
    list_of_models_for_selection.append(model_config.get_model_name())
    map_of_models_to_selection_list[model_config.get_model_name()] = i


def infer_sentiment_with_color_bar(review, model_choice):
    model_config = list_of_model_config[int(map_of_models_to_selection_list.get(model_choice))]
    model = map_of_models.get(model_config.get_model_module_name() + ":" + model_config.get_model_class_name(), None)
    logger.info(f"\n\nModel selected {model_choice}, model found? {model}")    

    ###
    ### If the user have not selected this model for inference before, then load the pre-trained model
    ###
    if model == None:
        logger.info(f"Model selected {model_choice}, model NOT FOUND.  Loading model {model_config}")    

        model = sa_model_pipeline.load_single_model(model_config.get_model_module_name(), 
                                                    model_config.get_model_class_name())
        map_of_models[model_config.get_model_module_name() + ":" + model_config.get_model_class_name()] = model
    
    ###
    ### Ask the model to infer the review
    ###
    sa_model_inference = model.inference(review)

    confidence = sa_model_inference.get_raw_prediction_value()
    confidence_percent = int(confidence * 100)
    filled = confidence_percent // 10
    empty = 10 - filled

    if confidence_percent < 25:
        fill_color = "#f44336"  # Red
    elif confidence_percent <= 90:
        fill_color = "#ff9800"  # Yellow
    else:
        fill_color = "#4CAF50"  # Green

    bar_html = f'''
    <div style="
        width: 110px;
        height: 20px;
        background-color: #ccc;
        border: 1px solid #bbb;
        position: relative;
    ">
        <div style="
            width: {confidence_percent}%;
            height: 100%;
            background: {fill_color};
            position: absolute;
            top: 0;
            left: 0;
        "></div>
    </div>
    '''
    text_color = "#4CAF50" if sa_model_inference.get_interpreted_prediction_str_value() == "Positive" else "#f44336"  # green or red

    html_output = f"""
    <b>{model_choice} → <span style='color:{text_color}'>{sa_model_inference.get_interpreted_prediction_str_value()}</span> (Raw %: {confidence_percent}%)</b><br>
    <div style="margin-top:4px;">{bar_html}</div>
    """
    return html_output

amazon_review_demo = gr.Interface(
    fn=infer_sentiment_with_color_bar,
    inputs=[
        gr.Textbox(lines=6, placeholder="Paste your review here...", label="Review Text"),
        gr.Dropdown(choices=list_of_models_for_selection, label="Choose Model")
    ],
    outputs=gr.HTML(label="Inference Result"),
    title="Amazon Review Sentiment Analysis/Classifier",
    description="Enter a review and choose a model to infer sentiment.",
    allow_flagging="never",
    live=False,
    submit_btn="Infer"
)

amazon_review_demo.launch()
