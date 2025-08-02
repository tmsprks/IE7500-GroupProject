import os
import numpy as np
import logging
import inspect

import tensorflow as tf

from keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, f1_score, classification_report

from model.SASentimentModel import SASentimentModel
from utils.kaggle_dataset import KaggleDataSet
from utils.sa_model_params import SAModelParams
from utils.sa_app_config import SAAppConfig
from utils.sa_model_inference import SAModelInference
from model.SASelfAttentionLayer import SASelfAttentionLayer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                                            "batch_size",
                                            "chkpt_file_ext"]
    
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
        self.chkpt_file_name_extension = model_params.get_model_param("chkpt_file_ext")

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

        logger.info(f"{class_name}.{method_name}(): X_train: {len(self.X_train)}, y_train: {len(self.y_train)}, X_test: {len(self.X_test)}, y_test: {len(self.y_test)}, X_val: {len(self.X_val)}, y_val: {len(self.y_val)}")

        ########################################################################
        ###
        ### NOTE: Right below we will create a TextVectorization component for the NN.
        ###       Therefore, we don't really need to create a tokenizer & pad sequences to train the model
        ###       because TextVectorization will handle any string input dynamically 
        ###       and create the correct sequencing for the network
        ###       Keeping all this code for now in case we want to remove the TextVectorization component
        ###
        self.tokenizer.fit_on_texts(self.X_train)
        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_val_seq = self.tokenizer.texts_to_sequences(self.X_val)
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        self.X_train_pad = pad_sequences(self.X_train_seq, maxlen=self.sequence_length, padding='post')
        self.X_val_pad = pad_sequences(self.X_val_seq, maxlen=self.sequence_length, padding='post')
        self.X_test_pad = pad_sequences(self.X_test_seq, maxlen=self.sequence_length, padding='post')
        ###
        ########################################################################

        ###
        ### Create the TextVectorization component for the NN
        ###
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
            ### Be sure that we are passing string as input to the NN instead of preprocessed/tokenized sequences
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
            ### Add recurrent dropout for regularization
            ### Output: (batch_size, 100, 128) (64 units × 2 for bidirectional)
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, 
                                     return_sequences=True,
                                     recurrent_dropout=0.3,
                                     dropout=0.2)
            ),  

            ### SelfAttention layer to compute context vector
            ### Output: (batch_size, 64*2=128), attention_weights
            SASelfAttentionLayer(units=64),  

            ### Dense layer with ReLU activation for feature extraction
            ### Add L2 regularization to combat potential overfitting
            ### Output: (batch_size, 64)
            tf.keras.layers.Dense(64, 
                                  activation='relu',
                                  kernel_regularizer=l2(0.001)),  

            ### Dropout layer for regularization
            ### Output: (batch_size, 64)
            tf.keras.layers.Dropout(0.5),  

            ### Add L2 regularization to the final Dense layer
            tf.keras.layers.Dense(1, 
                                  activation='sigmoid', 
                                  name='output',
                                  kernel_regularizer=l2(0.001))  
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info(f"Calling {class_name}.{method_name}(): Model compiled")

        logger.info(f"Calling {class_name}.{method_name}(): Fitting model: X_train: {len(self.X_train)}, y_train: {len(self.y_train)}, X_val: {len(self.X_val)}, y_val: {len(self.y_val)}")

        ###
        ### NOTE:  Since we are utilizing a TextVectorizer in the NN, be sure to pass X_train.values rather than any preprocessed/tokenized data like X_test_pad, etc
        ###
        self.history = self.model.fit(x=self.X_train[KaggleDataSet.get_review_column_name()].values, 
                                      y=self.y_train.values, 
                                      validation_data=(self.X_val[KaggleDataSet.get_review_column_name()].values, self.y_val.values), 
                                      epochs=int(sa_model_param.get_model_param("epoch")),  
                                      batch_size=int(sa_model_param.get_model_param("batch_size")), 
                                      verbose=1)
        logger.info(f"{class_name}.{method_name}(): Model fitted.")


    def predict(self, sa_model_param:SAModelParams=None) -> None:    
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.y_pred = self.model.predict(self.X_test[KaggleDataSet.get_review_column_name()].values)
        self.result = (self.y_pred > 0.5).astype(int).flatten()
   
        print(classification_report(self.y_test.values, self.result))
        logger.info(f"{class_name}.{method_name}(): Completed")

    def inference(self, text_to_make_prediction_on: str=None) -> SAModelInference:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}, '{text_to_make_prediction_on}'")

        ###
        ### Pass the text_to_make_prediction_on to the model's predict method without any preprocessing/tokenization
        ### since we have a TextVectorization component in the network that will take care of preprocessing, 
        ### tokenization and sequencing of the text to the appropriate padding, sequence length, etc
        ###
        ### Using p.array([text_to_make_prediction_on], dtype=object) because the tensorflow.keras layer expects a list of text
        ###
        y_pred = self.model.predict(np.array([text_to_make_prediction_on], dtype=object), verbose=0)
        result = (y_pred > 0.5).astype(int)

        ###
        ### Confirm the shape is what you expected from the model's output layer because we need to extract the prediction and result
        ###
        ###print("Y_pred shape: ", y_pred.shape)
        ###print("result shape: ", result.shape)

        ###
        ### Use shape agnostic y_pred.item() instead of double indexing in this case like y_pred[0][0] or result[0][0]
        ### to access the actual raw prediction and interpreted value
        ###
        return_value = SAModelInference(text_to_make_prediction_on, y_pred.item(), result.item())

        logger.info(f"{class_name}.{method_name}(): Prediction is {result.item()}, raw pred is {y_pred.item()}")
        return return_value

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


    def save(self) -> None:
        ### Save keras model to checkpoint

        module_name = inspect.getmodule(inspect.currentframe()).__name__
        class_name = self.__class__.__name__
        self.model.save(super().get_checkpoint_file_name(module_name, class_name, self.chkpt_file_name_extension))


    def load(self) -> None:
        ###
        ### Loads keras model from checkpoint
        ###
        module_name = inspect.getmodule(inspect.currentframe()).__name__
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_app_config()}")    

        model_checkpoint_path = super().get_checkpoint_file_name(module_name, class_name, self.chkpt_file_name_extension)
        logger.info(f"{class_name}.{method_name}(): Loading model from {model_checkpoint_path}")    

        ### 
        ### Load the model using the model_checkpoint_path and DO NOT RECOMPILE the model again
        ###
        self.model = tf.keras.models.load_model(model_checkpoint_path, 
                                                compile=False)
        
        logger.info(f"{class_name}.{method_name}(): Model loaded successfully from {model_checkpoint_path}")    
