import numpy as np
import logging
import inspect

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from SASentimentModel import SASentimentModel, SAModelParams
from kaggle_dataset import KaggleDataSet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAModelTemplate(SASentimentModel):
    def __init__(self, model_params: SAModelParams):
        super().__init__(model_params)
        self.model = None               ### store the actual model you will create later

    def register(self, sa_model_param:SAModelParams=None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        return_value = f"Calling {class_name}.{method_name}: {super().get_model_params()}"
        return return_value


    def preprocess(self, model_params: SAModelParams) -> None:

        X_train = model_params.get_train_df()
        y_train = model_params.get_train_df()[KaggleDataSet.POLARITY_COLUMN_NAME]
        X_val = model_params.get_validation_df()
        y_val = model_params.get_validation_df()[KaggleDataSet.POLARITY_COLUMN_NAME]
        X_test = model_params.get_test_df()
        y_test = model_params.get_test_df()[KaggleDataSet.POLARITY_COLUMN_NAME]

        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}: {super().get_model_params()}")
              


    def fit(self, sa_model_param:SAModelParams=None) -> None:

        ### Sample code of what you can do in the fit method
        ### You are responsible for all the parameters saving/modification using sa_model_params

        '''
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3, batch_size=32, verbose=1
        )
        '''
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}: {super().get_model_params()}")

    def predict(self, sa_model_param:SAModelParams=None) -> None:

        ### Sample code
        '''
        y_pred = self.model.predict(X_test)
        '''
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}: {super().get_model_params()}")


    def evaluate(self, sa_model_param:SAModelParams=None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}: {super().get_model_params()}")
