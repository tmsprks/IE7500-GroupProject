
import os
import numpy as np
import logging
import inspect
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from model.SASentimentModel import SASentimentModel
from utils.kaggle_dataset import KaggleDataSet
from utils.sa_model_params import SAModelParams
from utils.sa_app_config import SAAppConfig
from utils.sa_model_inference import SAModelInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SALSTMModel(SASentimentModel):

    _LSTM_MODEL_PARAMS_LIST = ["vocab_size", "sequence_length", "embedding_dim", 
                               "max_features", "epoch", "batch_size", "chkpt_file_ext"]

    def __init__(self, sa_app_config: SAAppConfig, model_params: SAModelParams):
        super().__init__(sa_app_config, model_params)
        self.vocab_size = int(model_params.get_model_param("vocab_size"))
        self.sequence_length = int(model_params.get_model_param("sequence_length"))
        self.embedding_dim = int(model_params.get_model_param("embedding_dim"))
        self.max_features = int(model_params.get_model_param("max_features"))
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.chkpt_file_name_extension = model_params.get_model_param("chkpt_file_ext")
        self.model = None
        self.vectorize_layer = None
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.y_pred, self.result = None, None

    def register(self, sa_model_param: SAModelParams = None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        sa_model_param.verify_model_params(self._LSTM_MODEL_PARAMS_LIST)
        return f"Registered {class_name} with params: {self._LSTM_MODEL_PARAMS_LIST}"

    def preprocess(self, model_params: SAModelParams) -> None:
        self.X_train = model_params.get_train_df()
        self.y_train = self.X_train[KaggleDataSet.get_polarity_column_name()]
        self.X_val = model_params.get_validation_df()
        self.y_val = self.X_val[KaggleDataSet.get_polarity_column_name()]
        self.X_test = model_params.get_test_df()
        self.y_test = self.X_test[KaggleDataSet.get_polarity_column_name()]

        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )
        self.vectorize_layer.adapt(self.X_train[KaggleDataSet.get_review_column_name()].values)

    def fit(self, sa_model_param: SAModelParams = None) -> None:
        self.model = tf.keras.Sequential([
            self.vectorize_layer,
            tf.keras.layers.Embedding(
                input_dim=self.embedding_dim,
                output_dim=128,
                input_length=self.sequence_length
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    64,
                    return_sequences=False,
                    dropout=0.3,           # Dropout between LSTM inputs
                    recurrent_dropout=0.3  # Added recurrent dropout
                )
            ),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.history = self.model.fit(
            x=self.X_train[KaggleDataSet.get_review_column_name()].values,
            y=self.y_train.values,
            validation_data=(
                self.X_val[KaggleDataSet.get_review_column_name()].values,
                self.y_val.values
            ),
            epochs=int(sa_model_param.get_model_param("epoch")),
            batch_size=int(sa_model_param.get_model_param("batch_size")),
            verbose=1
        )

    def predict(self, sa_model_param: SAModelParams = None) -> None:
        self.y_pred = self.model.predict(self.X_test[KaggleDataSet.get_review_column_name()].values)
        self.result = (self.y_pred > 0.5).astype(int).flatten()
        print(classification_report(self.y_test.values, self.result))

    def inference(self, text_to_make_prediction_on: str = None) -> SAModelInference:
        y_pred = self.model.predict(np.array([text_to_make_prediction_on], dtype=object), verbose=0)
        result = (y_pred > 0.5).astype(int)
        return SAModelInference(text_to_make_prediction_on, y_pred.item(), result.item())

    def evaluate(self, sa_model_param: SAModelParams = None) -> None:
        acc, loss = self.model.evaluate(
            x=self.X_test[KaggleDataSet.get_review_column_name()].values,
            y=self.y_test.values, verbose=1
        )
        logger.info(f"Evaluation - Accuracy: {acc}, Loss: {loss}")

    def summary(self, sa_model_param: SAModelParams = None) -> None:
        self.model.summary()

    def save(self) -> None:
        module_name = inspect.getmodule(inspect.currentframe()).__name__
        class_name = self.__class__.__name__
        self.model.save(super().get_checkpoint_file_name(module_name, class_name, self.chkpt_file_name_extension))

    def load(self) -> None:
        module_name = inspect.getmodule(inspect.currentframe()).__name__
        class_name = self.__class__.__name__
        model_checkpoint_path = super().get_checkpoint_file_name(module_name, class_name, self.chkpt_file_name_extension)
        self.model = tf.keras.models.load_model(model_checkpoint_path, compile=False)
