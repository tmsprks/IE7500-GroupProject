import numpy as np
import logging
import inspect

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score
from SASentimentModel import SASentimentModel, SAModelParams
from kaggle_dataset import KaggleDataSet

'''
vocab_size=1000,max_features=20000,sequence_length=100,embedding_dim=20000,embed_output=128,bi_lstm_input_dim=64,self_attention_input_dim=64,dense_1_layer_dim=64,dense_1_dropout=0.5,final_dense_dim=2"
'''

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
        return context_vector, attention_weights

class SASelfAttentionModel(SASentimentModel):
    def __init__(self, model_params: SAModelParams):
        super().__init__(model_params)
        self.vocab_size = model_params.get_model_param("vocab_size")
        self.max_length = model_params.get_model_param("sequence_length")
        self.embedding_dim = model_params.get_model_param("embedding_dim")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.model = None

    def register(self, sa_model_param:SAModelParams=None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        return_value = f"Calling {class_name}.{method_name}(): {super().get_model_params()}"
        return return_value


    def preprocess(self, model_params: SAModelParams) -> None:

        '''
        X_train = model_params.get_train_df()
        y_train = model_params.get_train_df()[KaggleDataSet.POLARITY_COLUMN_NAME]
        X_val = model_params.get_validation_df()
        y_val = model_params.get_validation_df()[KaggleDataSet.POLARITY_COLUMN_NAME]
        X_test = model_params.get_test_df()
        y_train = model_params.get_test_df()[KaggleDataSet.POLARITY_COLUMN_NAME]
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_length, padding='post')
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_length, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_length, padding='post')
        return X_train_pad, y_train, X_val_pad, y_val, X_test_pad
        '''
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")
              


    def fit(self, sa_model_param:SAModelParams=None) -> None:
        '''
        inputs = tf.keras.Input(shape=(self.max_length,))
        embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length)(inputs)
        attention_output, _ = SASelfAttentionLayer(units=self.embedding_dim)(embedding)
        dense = tf.keras.layers.Dense(32, activation='relu')(attention_output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
        
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
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

    def predict(self, sa_model_param:SAModelParams=None) -> None:
        '''
        y_pred = self.model.predict(X_test)
        return (y_pred > 0.5).astype(int).flatten()
        '''
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")


    def evaluate(self, sa_model_param:SAModelParams=None) -> None:
        '''
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        '''
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")
