import os
import pickle
import logging
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam

from model.SASentimentModel import SASentimentModel
from utils.kaggle_dataset import KaggleDataSet
from utils.sa_model_params import SAModelParams
from utils.sa_app_config import SAAppConfig
from utils.sa_model_inference import SAModelInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SARnnModel(SASentimentModel):

    _RNN_MODEL_PARAMS_LIST = ["vocab_size",
                              "sequence_length",
                              "embedding_dim",
                              "epoch",
                              "batch_size",
                              "learning_rate",
                              "chkpt_file_ext"
                              ]

    def __init__(self, 
                sa_app_config: SAAppConfig,
                model_params: SAModelParams):
        super().__init__(sa_app_config, model_params)
        self.vocab_size = int(model_params.get_model_param("vocab_size"))
        self.sequence_length = int(model_params.get_model_param("sequence_length"))
        self.embedding_dim = int(model_params.get_model_param("embedding_dim"))
        self.epoch = int(model_params.get_model_param("epoch"))
        self.batch_size = int(model_params.get_model_param("batch_size"))
        self.learning_rate = float(model_params.get_model_param("learning_rate"))
        self.chkpt_file_ext = str(model_params.get_model_param("chkpt_file_ext"))

        ### Tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.model = None
        self.history = None

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        self.X_train_pad = self.X_val_pad = self.X_test_pad = None
        self.result = None

    def register(self, sa_model_param: SAModelParams = None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        sa_model_param.verify_model_params(SARnnModel._RNN_MODEL_PARAMS_LIST)
        logger.info(f"{class_name}.{method_name}(): Completed")
        return f"Registered model with parameters: {super().get_model_params()}"

    def preprocess(self, model_params: SAModelParams) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        self.X_train = model_params.get_train_df()
        self.y_train = self.X_train[KaggleDataSet.get_polarity_column_name()].replace({2: 1})

        self.X_val = model_params.get_validation_df()
        self.y_val = self.X_val[KaggleDataSet.get_polarity_column_name()].replace({2: 1})

        self.X_test = model_params.get_test_df()
        self.y_test = self.X_test[KaggleDataSet.get_polarity_column_name()].replace({2: 1})

        ### Fit tokenizer
        self.tokenizer.fit_on_texts(self.X_train[KaggleDataSet.get_review_column_name()])
        
        ### Pad seqeuence & length
        self.X_train_pad = pad_sequences(
            self.tokenizer.texts_to_sequences(self.X_train[KaggleDataSet.get_review_column_name()]),
            maxlen=self.sequence_length, padding='post'
        )
        self.X_val_pad = pad_sequences(
            self.tokenizer.texts_to_sequences(self.X_val[KaggleDataSet.get_review_column_name()]),
            maxlen=self.sequence_length, padding='post'
        )
        self.X_test_pad = pad_sequences(
            self.tokenizer.texts_to_sequences(self.X_test[KaggleDataSet.get_review_column_name()]),
            maxlen=self.sequence_length, padding='post'
        )


        logger.info(f"{class_name}.{method_name}(): Completed")


    def fit(self, sa_model_param: SAModelParams = None, resume_training=False, initial_epoch=0) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        checkpoint_base_path = self.get_checkpoint_file_name(
            inspect.getmodule(inspect.currentframe()).__name__, self.__class__.__name__
        ) + self.chkpt_file_ext

        if os.path.exists(checkpoint_base_path):
            logger.info(f"{class_name}.{method_name}(): Loading model from checkpoint: {checkpoint_base_path}")
            self.model = tf.keras.models.load_model(checkpoint_base_path)
            logger.info(f"{class_name}.{method_name}(): Model loaded successfully.")
            return  # stops resume training
        else:
            logger.info(f"{class_name}.{method_name}(): No checkpoint found at {checkpoint_base_path}. Training new model..")

        if self.model is None:
            # builds and compiles model if nothing has been loaded from checkpoint
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim),
                tf.keras.layers.SimpleRNN(64, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.SimpleRNN(64),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            logger.info(f"{class_name}.{method_name}(): Model compiled")

        checkpoint_dir = self.sa_app_config.model.checkpoint_dir
        checkpoint_file_prefix = os.path.join(checkpoint_dir)
        checkpoint_file_prefix += '\\'
        logger.info(f"{class_name}.{method_name}(): Checkpoint dir: {checkpoint_dir}.   Checkpoint path: {checkpoint_file_prefix}")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_checkpoint_file_name(
                inspect.getmodule(inspect.currentframe()).__name__, self.__class__.__name__
            ) + "_epoch{epoch:02d}_valacc{val_accuracy:.4f}.keras",
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

        self.model.save(checkpoint_base_path)
        logger.info(f"{class_name}.{method_name}(): Model fitted")

    def predict(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"Calling {class_name}.{method_name}(): {super().get_model_params()}")

        y_pred = self.model.predict(self.X_test_pad)
        self.result = (y_pred > 0.5).astype(int).flatten()

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

    def save(self) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        module_name = inspect.getmodule(inspect.currentframe()).__name__

        # base path no extension
        base_checkpoint_path = super().get_checkpoint_file_name(module_name, class_name)

        keras_checkpoint_path = base_checkpoint_path + ".keras"

        self.model.save(keras_checkpoint_path)
        logger.info(f"{class_name}.{method_name}(): Model saved to {keras_checkpoint_path}")

        tokenizer_path = base_checkpoint_path + "_tokenizer.pickle"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        logger.info(f"{class_name}.{method_name}(): Tokenizer saved to {tokenizer_path}")

    def load(self) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        module_name = inspect.getmodule(inspect.currentframe()).__name__

        base_checkpoint_path = super().get_checkpoint_file_name(module_name, class_name)
        keras_checkpoint_path = base_checkpoint_path + ".keras"

        logger.info(f"{class_name}.{method_name}(): Loading model from {keras_checkpoint_path}")

        self.model = tf.keras.models.load_model(keras_checkpoint_path, compile=False)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        logger.info(f"{class_name}.{method_name}(): Model loaded and compiled")

        tokenizer_path = base_checkpoint_path + "_tokenizer.pickle"
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            logger.info(f"{class_name}.{method_name}(): Tokenizer loaded from {tokenizer_path}")
        else:
            logger.warning(f"{class_name}.{method_name}(): Tokenizer file not found. You may need to re-fit it.")

    def inference(self, text_to_make_prediction_on: str = None) -> SAModelInference:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")

        # preprocess text to sequences
        sequences = self.tokenizer.texts_to_sequences([text_to_make_prediction_on])
        padded_seq = pad_sequences(
            sequences, 
            maxlen=self.sequence_length, 
            padding='post'
        )

        # predict raw probability
        probability = self.model.predict(padded_seq, verbose=0)[0][0]
    
        pred = int(probability >= 0.5)

        return SAModelInference(
            prediction_text=text_to_make_prediction_on,
            raw_prediction=probability,
            interpreted_prediction=pred
    )