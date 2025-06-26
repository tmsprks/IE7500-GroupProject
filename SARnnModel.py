import os
import numpy as np
import logging
import inspect
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from SASentimentModel import SASentimentModel
from kaggle_dataset import KaggleDataSet
from sa_model_params import SAModelParams

import re
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SARnnModel(SASentimentModel):

    _RNN_MODEL_PARAMS_LIST = ["vocab_size",
                              "sequence_length",
                              "embedding_dim",
                              "epoch",
                              "batch_size"]

    def __init__(self, model_params: SAModelParams):
        super().__init__(model_params)
        self.vocab_size = int(model_params.get_model_param("vocab_size"))
        self.sequence_length = int(model_params.get_model_param("sequence_length"))
        self.embedding_dim = int(model_params.get_model_param("embedding_dim"))
        self.epoch = int(model_params.get_model_param("epoch"))
        self.batch_size = int(model_params.get_model_param("batch_size"))

        ### Tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        self.X_train_seq = self.X_val_seq = self.X_test_seq = None
        self.X_train_pad = self.X_val_pad = self.X_test_pad = None

        self.model = None
        self.y_pred = None
        self.result = None

    def register(self, sa_model_param: SAModelParams = None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        return_value = f"Calling {class_name}.{method_name}(): {super().get_model_params()}"
        logger.info(return_value)

        sa_model_param.verify_model_params(SARnnModel._RNN_MODEL_PARAMS_LIST)
        logger.info(f"{class_name}.{method_name}(): Completed")
        return return_value

    def preprocess(self, model_params: SAModelParams) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.X_train = model_params.get_train_df()
        self.y_train = self.X_train[KaggleDataSet.get_polarity_column_name()]

        self.X_val = model_params.get_validation_df()
        self.y_val = self.X_val[KaggleDataSet.get_polarity_column_name()]

        self.X_test = model_params.get_test_df()
        self.y_test = self.X_test[KaggleDataSet.get_polarity_column_name()]

        ### Normalize labels: change label '2' to '1' for binary classifiation
        self.y_train = self.y_train.replace({2: 1})
        self.y_val = self.y_val.replace({2: 1})
        self.y_test = self.y_test.replace({2: 1})

        ### Fit tokenizer
        self.tokenizer.fit_on_texts(self.X_train[KaggleDataSet.get_review_column_name()])
        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train[KaggleDataSet.get_review_column_name()])
        self.X_val_seq = self.tokenizer.texts_to_sequences(self.X_val[KaggleDataSet.get_review_column_name()])
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test[KaggleDataSet.get_review_column_name()])

        ### Pad seqeuence & length
        self.X_train_pad = pad_sequences(self.X_train_seq, maxlen=self.sequence_length, padding='post')
        self.X_val_pad = pad_sequences(self.X_val_seq, maxlen=self.sequence_length, padding='post')
        self.X_test_pad = pad_sequences(self.X_test_seq, maxlen=self.sequence_length, padding='post')

        logger.info(f"{class_name}.{method_name}(): Completed")

    def load_model_from_file(self, checkpoint_path: str) -> None:
        ### Loads keras model from checkpoint
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = tf.keras.models.load_model(checkpoint_path, compile=False)
        # Recompile optimzer
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),  # can change optimzer/learning rate **Just make sure you import the optimizer
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model loaded and compiled successfully")

    def fit(self, sa_model_param: SAModelParams = None, resume_training=False, initial_epoch=0) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Starting fit; resume_training={resume_training}, initial_epoch={initial_epoch}")

        if self.model is None:
            # builds and compiles model if nothing has been loaded from checkpoint
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim
                ),
                tf.keras.layers.SimpleRNN(64, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.SimpleRNN(64),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            logger.info("Built and compiled new model")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath='model_checkpoint_epoch{epoch:02d}_valacc{val_accuracy:.4f}.keras',
            monitor='val_accuracy',
            save_best_only=False, ### Saves every epoch
            save_weights_only=False, ### Saves entire model
            save_freq='epoch',
            verbose=1
        )

        ### Fit model padded sequences
        self.history = self.model.fit(
            x=self.X_train_pad,
            y=self.y_train.values,
            validation_data=(self.X_val_pad, self.y_val.values),
            epochs=self.epoch,
            batch_size=self.batch_size,
            callbacks=[checkpoint_cb],
            initial_epoch=initial_epoch if resume_training else 0,
            verbose=1
        )

        ### Logs best validation accurary
        best_epoch = max(self.history.epoch) + 1
        best_val_acc = max(self.history.history['val_accuracy'])
        logger.info(f"Best val_accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

        logger.info(f"{class_name}.{method_name}(): Model fitted")
        logger.info(f"{class_name}.{method_name}(): Completed")

    def predict(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.y_pred = self.model.predict(self.X_test_pad)
        self.result = (self.y_pred > 0.5).astype(int).flatten()

        print(classification_report(self.y_test.values, self.result))
        logger.info(f"{class_name}.{method_name}(): Completed")

    def evaluate(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        loss, accuracy = self.model.evaluate(self.X_test_pad, self.y_test.values, verbose=1)
        logger.info(f"{class_name}.{method_name}(): Accuracy: {accuracy}, Loss: {loss}")
        logger.info(f"{class_name}.{method_name}(): Completed")

    def summary(self, sa_model_param: SAModelParams = None) -> None:
        self.model.summary()