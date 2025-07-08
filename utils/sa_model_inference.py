
import os
import logging
import inspect
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelInference:

    _SA_POLARITY_VALUE_0 = 0
    _SA_POLARITY_VALUE_1 = 1

    _SA_POLARLITY_POSITIVE_STR_VAL = "Positive"
    _SA_POLARLITY_NEGATIVE_STR_VAL = "Negative"

    _SA_POLARITY_PRED_VAL_TO_STR_VAL = {_SA_POLARITY_VALUE_0: _SA_POLARLITY_NEGATIVE_STR_VAL,
                                        _SA_POLARITY_VALUE_1: _SA_POLARLITY_POSITIVE_STR_VAL}

    @staticmethod
    def interpret_pred_value_to_string_value(interpret_pred_value_to_convert: int) -> str:
        return SAModelInference._SA_POLARITY_PRED_VAL_TO_STR_VAL.get(interpret_pred_value_to_convert, "UNKNOWN PREDICTION VALUE")
    
    def __init__(self, 
                 prediction_text:str,
                 raw_prediction:float,
                 interpreted_prediction:int):
        self.prediction_text = prediction_text
        self.raw_prediction = raw_prediction
        self.interpreted_prediction = interpreted_prediction

        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): '{self.prediction_text}', {self.raw_prediction}, {self.interpreted_prediction}")       

    def __str__(self):
        return f"'{self.get_prediction_text()}', {self.get_raw_prediction_value()} -> {SAModelInference.interpret_pred_value_to_string_value(self.get_interpreted_prediction())}"
    
    def get_prediction_text(self) -> str:
        return self.prediction_text
    
    def get_raw_prediction_value(self) -> float:
        return self.raw_prediction
    
    def get_interpreted_prediction(self) -> int:
        return self.interpreted_prediction
