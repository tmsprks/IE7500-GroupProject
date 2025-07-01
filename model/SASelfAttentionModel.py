import os
import numpy as np
import logging
import inspect

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, classification_report

from model.SASentimentModel import SASentimentModel
from utils.kaggle_dataset import KaggleDataSet

from utils.sa_model_config_loader import SAModelConfigLoader
from utils.sa_data_loader import SADataLoader
from utils.sa_model_params import SAModelParams
from utils.sa_app_config import SAAppConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SASelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SASelfAttentionLayer, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        ###return context_vector, attention_weights
        return context_vector

class SASelfAttentionModel(SASentimentModel):

    ### 
    ### List all the model parameters the model will be using here
    ### Pass this list to the SAModelParams.verify_model_params method in the register method
    ###

    _SA_SELF_ATTENTION_MODEL_PARAMS_LIST = ["vocab_size",
                                            "sequence_length",
                                            "embedding_dim",
                                            "max_features",
                                            "epoch",
                                            "batch_size"]
    
    def __init__(self, 
                 sa_app_config: SAAppConfig,
                 model_params: SAModelParams):
        super().__init__(sa_app_config, model_params)
        self.vocab_size = int(model_params.get_model_param("vocab_size"))
        self.sequence_length = int(model_params.get_model_param("sequence_length"))
        self.embedding_dim = int(model_params.get_model_param("embedding_dim"))
        self.max_features = int(model_params.get_model_param("max_features"))
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.result = None
        self.X_train_seq = None
        self.X_val_seq = None
        self.X_test_seq = None
        self.X_train_pad = None
        self.X_val_pad = None
        self.X_test_pad = None
        self.model = None

    def register(self, sa_model_param:SAModelParams=None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        return_value = f"Calling {class_name}.{method_name}(): {super().get_model_params()}"
        logger.info(return_value)

        ### 
        ### Check, before we run the model, that the model params the model will be using are defined
        ### in the model config entry for this model.
        ###
        sa_model_param.verify_model_params(SASelfAttentionModel._SA_SELF_ATTENTION_MODEL_PARAMS_LIST)
        logger.info(f"{class_name}.{method_name}(): Completed")
    
        return return_value


    def preprocess(self, model_params: SAModelParams) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.X_train = model_params.get_train_df()
        self.y_train = model_params.get_train_df()[KaggleDataSet.get_polarity_column_name()]
        self.X_val = model_params.get_validation_df()
        self.y_val = model_params.get_validation_df()[KaggleDataSet.get_polarity_column_name()]
        self.X_test = model_params.get_test_df()
        self.y_test = model_params.get_test_df()[KaggleDataSet.get_polarity_column_name()]
        self.tokenizer.fit_on_texts(self.X_train)
        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_val_seq = self.tokenizer.texts_to_sequences(self.X_val)
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        self.X_train_pad = pad_sequences(self.X_train_seq, maxlen=self.sequence_length, padding='post')
        self.X_val_pad = pad_sequences(self.X_val_seq, maxlen=self.sequence_length, padding='post')
        self.X_test_pad = pad_sequences(self.X_test_seq, maxlen=self.sequence_length, padding='post')

        ### Create the TextVectorization layer
        self.vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=self.max_features,
                                                                 output_mode='int',
                                                                 output_sequence_length=self.sequence_length)

        self.vectorize_layer.adapt(self.X_train[KaggleDataSet.get_review_column_name()].values)
        logger.info(f"{class_name}.{method_name}(): Completed")


              
    def fit(self, sa_model_param:SAModelParams=None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.model = tf.keras.Sequential([
            ### TextVectorization layer to convert text to integer sequences
            ### Input: (batch_size, 1), Output: (batch_size, sequence_length=100 from above)
            self.vectorize_layer,  

            ### Embedding layer to map tokens to dense vectors
            ### Output: (batch_size, 100, 128)
            tf.keras.layers.Embedding(
                input_dim=self.embedding_dim,  # Vocabulary size
                output_dim=128,   # Embedding dimension
                input_length=100  # Sequence length from vectorize_layer
            ),  

            ### Bidirectional LSTM layer to capture sequential dependencies
            ### Output: (batch_size, 100, 128) (64 units × 2 for bidirectional)
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            ),  

            ### SelfAttention layer to compute context vector
            ### Output: (batch_size, 64*2=128), attention_weights
            SASelfAttentionLayer(units=64),  

            ### Dense layer with ReLU activation for feature extraction
            ### Output: (batch_size, 64)
            tf.keras.layers.Dense(64, activation='relu'),  

            ### Dropout layer for regularization
            ### Output: (batch_size, 64)
            tf.keras.layers.Dropout(0.5),  

            tf.keras.layers.Dense(1, activation='sigmoid', name='output')  
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info(f"Calling {class_name}.{method_name}(): Model compiled")

        logger.info(f"Calling {class_name}.{method_name}(): Fitting model: X_train: {len(self.X_train)}, y_train: {len(self.y_train)}, X_val: {len(self.X_val)}, y_val: {len(self.y_val)},")
        self.history = self.model.fit(x=self.X_train[KaggleDataSet.get_review_column_name()].values, 
                                      y=self.y_train.values, 
                                      validation_data=(self.X_val[KaggleDataSet.get_review_column_name()].values, self.y_val.values), 
                                      epochs=int(sa_model_param.get_model_param("epoch")),  
                                      batch_size=int(sa_model_param.get_model_param("batch_size")), 
                                      verbose=1)
        logger.info(f"Calling {class_name}.{method_name}(): Model fitted")
        logger.info(f"{class_name}.{method_name}(): Completed")


    def predict(self, sa_model_param:SAModelParams=None) -> None:    
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.y_pred = self.model.predict(self.X_test[KaggleDataSet.get_review_column_name()].values)
        self.result = (self.y_pred > 0.5).astype(int).flatten()
   
        print(classification_report(self.y_test.values, self.result))
        logger.info(f"{class_name}.{method_name}(): Completed")


    def evaluate(self, sa_model_param:SAModelParams=None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")
        
        accuracy, loss = self.model.evaluate(
            x=self.X_test[KaggleDataSet.get_review_column_name()].values,           
            y=self.y_test.values,      
            verbose=1             
        )
        logger.info(f"{class_name}.{method_name}(): Completed")


    def summary(self, sa_model_param:SAModelParams=None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")        

        self.model.summary ()
        logger.info(f"{class_name}.{method_name}(): Completed")

