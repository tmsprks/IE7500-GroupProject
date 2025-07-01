import os
import sys
import logging

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)
print("Project directory added:", project_dir)

# Debug utils import
try:
    import utils
    print("utils package found at:", utils.__file__)
except ImportError as e:
    print("Import error for utils:", e)

from utils.kaggle_dataset import KaggleDataSet
from utils.sa_data_loader import SADataLoader
from utils.sa_app_config_loader import SAAppConfigLoader

print("Imported all utils modules successfully")


train_file = r"data\train_50K.csv"
test_file = r"data\test_10K.csv"
validate_file = r"data\validate_10K.csv"

cwd = os.getcwd()
path_to_train_csv_file = os.path.join(cwd, train_file)
path_to_test_csv_file = os.path.join(cwd, test_file)
path_to_validate_csv_file = os.path.join(cwd, validate_file)

sa_data_loader = SADataLoader(SAAppConfigLoader().get_app_config(), path_to_train_csv_file, path_to_test_csv_file, path_to_validate_csv_file)
sa_data_loader.load_data(KaggleDataSet.get_kaggle_column_names())

train_df = sa_data_loader.get_train_df()
test_df = sa_data_loader.get_test_df()
validate_df = sa_data_loader.get_validation_df()


print("Unique Kaggle label counts.  Unique values should be 1s and 2s")
print(train_df[KaggleDataSet.get_kaggle_polarity_column_name()].value_counts())
print(test_df[KaggleDataSet.get_kaggle_polarity_column_name()].value_counts())
print(validate_df[KaggleDataSet.get_kaggle_polarity_column_name()].value_counts())


print("Unique transformed label counts.  Unique values should be 0s and 1s")
print(train_df[KaggleDataSet.get_polarity_column_name()].value_counts())
print(test_df[KaggleDataSet.get_polarity_column_name()].value_counts())
print(validate_df[KaggleDataSet.get_polarity_column_name()].value_counts())