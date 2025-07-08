import os
import sys
import logging
import pprint

# Add project/ to sys.path (parent of test/)
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from utils.sa_app_config_loader import SAAppConfigLoader
from utils.sa_app_config import SAAppConfig
from utils.sa_data_factory import SADataFactory
from utils.sa_data_generator import SADataGenerator
from utils.kaggle_data_loader import KaggleDataLoader
from utils.kaggle_dataset import KaggleDataSet

print("Imported all modules successfully")

kaggle_path = "kritanjalijain/amazon-reviews"
kaggle_data_loader = KaggleDataLoader(kaggle_path)
kaggle_data_loader.load_data(KaggleDataSet.get_kaggle_column_names())

kaggle_dataset = KaggleDataSet(kaggle_data_loader)

train_label_count = kaggle_dataset.get_train_value_count()
test_label_count = kaggle_dataset.get_test_value_count()
print("Train label count:\n", train_label_count)
print("Test label count:\n", test_label_count)

sa_app_config_loader = SAAppConfigLoader()
sa_app_config = sa_app_config_loader.get_app_config()
sa_data_factory = SADataFactory(kaggle_dataset)   
generator = SADataGenerator(sa_data_factory)

test_config_dir = sa_app_config.test.config_dir
test_output_dir = sa_app_config.test.output_dir

results = generator.generate_datasets(test_config_dir,
                                      test_output_dir,
                                      ensure_uniqueness_across_same_dataset_type=True, 
                                      randomize_samples=False)
print("\n\nFile generations completed")

print(f"Number of data files generated: {len(results)}\n")
for output_file, ensure_uniqueness, randomize_samples, df in results:
    print(f"\nGenerated {output_file}, ensure uniqueness {ensure_uniqueness}, randomize samples {randomize_samples} with {len(df)} rows")
    print(df.head(2))