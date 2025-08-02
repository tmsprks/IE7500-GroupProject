

import os
import logging
import inspect
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

from model.SASentimentModel import SASentimentModel
from utils.kaggle_dataset import KaggleDataSet
from utils.sa_model_params import SAModelParams
from utils.sa_app_config import SAAppConfig
from utils.sa_model_inference import SAModelInference

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
###logger.setLevel(logging.CRITICAL)

###
### SAxyzBERTSentimentModel implements the pre-trained model defined in the model config file
### Ex: prajjwal1/bert-mini or prajjwal1/bert-tiny or other HF pre-trained-BERT model
###

class SAxyzBERTSentimentModel(SASentimentModel):
    _SA_MODEL_PARAMS_LIST = ["max_seq_length", "epoch", "batch_size", "pretrained_BERT_model", "learning_rate"]

    def __init__(self, sa_app_config: SAAppConfig, sa_model_param: SAModelParams = None):
        super().__init__(sa_app_config, sa_model_param)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        model_params = sa_model_param if sa_model_param else self.sa_model_params
        self.pretrained_BERT_model_name = model_params.get_model_param("pretrained_BERT_model")
        self.max_length = int(model_params.get_model_param("max_seq_length"))
        self.batch_size = int(model_params.get_model_param("batch_size"))
        self.learning_rate = float(model_params.get_model_param("learning_rate"))
        self.epochs = int(model_params.get_model_param("epoch"))
        
        ###
        ### Create the correct tokenizer & auto-model from HF from the model specified in the model config file
        ###
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_BERT_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_BERT_model_name, num_labels=2)
        self.model.to(self.device)

    def register(self, sa_model_param: SAModelParams = None) -> str:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        return_value = f"{class_name}.{method_name}(): {super().get_model_params()}"
        logger.info(return_value)

        ###
        ### Check, before we run the model, that the model params the model will be using are defined
        ### in the model config entry for this model.
        ###
        sa_model_param.verify_model_params(SAxyzBERTSentimentModel._SA_MODEL_PARAMS_LIST)
        logger.info(f"{class_name}.{method_name}(): Completed")
    
        return return_value

    def preprocess(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        model_params = sa_model_param if sa_model_param else self.sa_model_params
        self.X_train = model_params.get_train_df()
        self.y_train = self.X_train[KaggleDataSet.get_polarity_column_name()].values
        self.X_val = model_params.get_validation_df()
        self.y_val = self.X_val[KaggleDataSet.get_polarity_column_name()].values
        self.X_test = model_params.get_test_df()
        self.y_test = self.X_test[KaggleDataSet.get_polarity_column_name()].values
        logger.info(f"{class_name}.{method_name}(): X_train: {len(self.X_train)}, y_train: {len(self.y_train)}, X_test: {len(self.X_test)}, y_test: {len(self.y_test)}, X_val: {len(self.X_val)}, y_val: {len(self.y_val)}")
        
        review_col = KaggleDataSet.get_review_column_name()
        self.train_texts = self.X_train[review_col].values
        self.val_texts = self.X_val[review_col].values
        self.test_texts = self.X_test[review_col].values
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_length)
        
        train_dataset = Dataset.from_dict({"text": self.train_texts, "label": self.y_train})
        val_dataset = Dataset.from_dict({"text": self.val_texts, "label": self.y_val})
        test_dataset = Dataset.from_dict({"text": self.test_texts, "label": self.y_test})
        
        self.tokenized_train = train_dataset.map(tokenize_function, batched=True)
        self.tokenized_val = val_dataset.map(tokenize_function, batched=True)
        self.tokenized_test = test_dataset.map(tokenize_function, batched=True)
        
        self.tokenized_train = self.tokenized_train.remove_columns(["text"])
        self.tokenized_train = self.tokenized_train.rename_column("label", "labels")
        self.tokenized_train.set_format("torch")
        
        self.tokenized_val = self.tokenized_val.remove_columns(["text"])
        self.tokenized_val = self.tokenized_val.rename_column("label", "labels")
        self.tokenized_val.set_format("torch")
        
        self.tokenized_test = self.tokenized_test.remove_columns(["text"])
        self.tokenized_test = self.tokenized_test.rename_column("label", "labels")
        self.tokenized_test.set_format("torch")

    def fit(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        ### 
        ### Setup all the training parameters
        ###
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none"
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            return {"accuracy": acc}
        
        ###
        ### Utilizing HF's Trainder pipeline to tune the pre-trained model
        ### Abstract away the gradient calculation, backprop, etc
        ###
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_val,
            compute_metrics=compute_metrics
        )

        ###
        ### Start the training
        ###
        
        self.trainer.train()

    def predict(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        predictions = self.trainer.predict(self.tokenized_test)
        self.test_preds = np.argmax(predictions.predictions, axis=-1)

    def inference(self, text_to_make_prediction_on: str) -> SAModelInference:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        ###
        ### Turn off training mode so dropouts  & normalization are not active
        ### Otherwise we could get random behaviors if dropouts & normalization (which are used for those purposes during training) kicks in
        ###
        self.model.eval()

        ###
        ### Tokenize the text, prep it for the BERT model
        ### Ensure the tokenizer returns Pytorch tensors with return_tensors='pt'
        ### for compatibility with the Pytorch model
        ###
        encoding = self.tokenizer(text_to_make_prediction_on, 
                                  truncation=True, 
                                  padding=True, 
                                  max_length=self.max_length, 
                                  return_tensors='pt')
        
        ###
        ### Just in case the tensors are on the GPU, move everything to the same device
        ### Otherwise we get a run time error.  If not using GPU, then no harm is done
        ###
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        ###
        ### Explicitly turn off gradient during inference
        ### Gradients are only necessary during training for backprop', not during inference
        ### This saves memories & compute units
        ###
        with torch.no_grad():  
            ### Forward pass, passing in the input_ids and attention_mask
            ### Output is a SequenceClassifierOutput object with raw prediction
            outputs = self.model(**encoding)

        ### Extract the logits
        logits = outputs.logits

        ### Use argmax to convert raw logits
        pred = torch.argmax(logits, dim=1).item()
        ####sentiment = 'positive' if pred == 1 else 'negative'

        ### Normalize the logis into probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        ### Now find the raw confidence base on the predicted class, if pred=1, then probs[1].
        probability = probs[pred]
        
        return SAModelInference(prediction_text=text_to_make_prediction_on, raw_prediction=probability, interpreted_prediction=pred)

    def evaluate(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        eval_results = self.trainer.evaluate(self.tokenized_test)
        logger.info(f"Test Evaluation Results: {eval_results}")
        
        acc = accuracy_score(self.y_test, self.test_preds)
        report = classification_report(self.y_test, self.test_preds, target_names=['negative', 'positive'])
        logger.info(f"Test Accuracy: {acc}")
        logger.info(f"Classification Report:\n{report}")

    def summary(self, sa_model_param: SAModelParams = None) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        ###
        ### Print out the string version of the model
        ### HF models don't have a "summary()" method
        ###
        logger.info(str(self.model))

    def load(self) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        module_name = inspect.getmodule(inspect.currentframe()).__name__
        model_checkpoint_path = super().get_checkpoint_directory(module_name, class_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_BERT_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path, num_labels=2)
        self.model.to(self.device)

    def save(self) -> None:
        class_name = self.__class__.__name__
        method_name = inspect.currentframe().f_code.co_name
        logger.info(f"{class_name}.{method_name}(): {super().get_model_params()}")
        
        module_name = inspect.getmodule(inspect.currentframe()).__name__

        ###
        ### HF requires a directory & not a file name to save the model
        ###
        model_checkpoint_path = super().get_checkpoint_directory(module_name, class_name)

        ###
        ### Save the model & tokenizer separately
        ###
        self.model.save_pretrained(model_checkpoint_path)
        self.tokenizer.save_pretrained(model_checkpoint_path)