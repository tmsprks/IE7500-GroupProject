
from typing import List, Tuple
import pandas as pd
from kaggle_dataset import KaggleDataSet

class DataGenerator:

    HALF_SIZE = 2
    RANDOMIZE_STATE_VAL = 128

    ### 
    ### Ensure the GEN_DATA_INFO values match the ones in the gen_data_info.csv file
    ###
    GEN_DATA_INFO_FILE_NAME = "gen_data_info.csv"
    GEN_DATA_INFO_DATA_TYPE_COLUMN_NAME = "Data_Type"
    GEN_DATA_INFO_SAMPLE_SIZE_COLUMN_NAME = "Sample_Size"
    GEN_DATA_INFO_OUTPUT_FILE_COLUMN_NAME = "Output_File"
    GEN_DATA_INFO_DATASET_TYPE_TRAIN_VAL = "train"
    GEN_DATA_INFO_DATASET_TYPE_TEST_VAL = "test"
    GEN_DATA_INFO_DATASET_TYPE_VALIDATION_VAL = "validation"


    def __init__(self, 
                 kaggle_dataset: KaggleDataSet):

        ### Maps data set type to the dataframe variable
        self.dataset_map = {
            DataGenerator.GEN_DATA_INFO_DATASET_TYPE_TRAIN_VAL: kaggle_dataset.get_train_df(),
            DataGenerator.GEN_DATA_INFO_DATASET_TYPE_TEST_VAL: kaggle_dataset.get_test_df(),
            DataGenerator.GEN_DATA_INFO_DATASET_TYPE_VALIDATION_VAL: kaggle_dataset.get_validation_df()
        }

        ### Initialize a dictionary to track used indices for each dataset type
        ### in order to ensure uniqueness
        self.used_indices = {
            DataGenerator.GEN_DATA_INFO_DATASET_TYPE_TRAIN_VAL: set(),
            DataGenerator.GEN_DATA_INFO_DATASET_TYPE_TEST_VAL: set(),
            DataGenerator.GEN_DATA_INFO_DATASET_TYPE_VALIDATION_VAL: set()
        }


    def generate_datasets(self, 
                          info_file: str=None, 
                          ensure_uniqueness_across_same_dataset_type: bool=True, 
                          randomize_samples=True) -> List[Tuple[str, bool, bool, pd.DataFrame]]:

        """ generate_datasets downsamples the datasets from source DataFrames based on a configuration CSV.
            the source DataFrames are passed to the constructor.
        
            ensure_uniqueness_across_same_dataset_type: ensures the rows do not overlap across the same dataset type files.
                    In other words, row 1 will not appear in both train_10K.csv and train_100K.csv
            
            randomize_samples: determines whether the row sampling is deterministic or non-deterministic
                    If set to False, each subsequent call to generate_dataset may not return the same samples.  
                    If set to True, then the same samples are return after each call.
            
            ensure_uniqueness_across_same_dataset_type = True, randomize_samples = False:
                the behavior is no duplicate rows in the same dataset type files.
                the file will look the same after repeated calls to generate_dataset
                use this setting to generate files that don't overlap but deterministic
            
            ensure_uniqueness_across_same_dataset_type = True, randomize_samples = True
                the behavior is no duplicate rows in the same dataset type files.
                the file will not look the same after repeated calls to generate_dataset
                use this setting to generate files that don't overlap and non-deterministic

            ensure_uniqueness_across_same_dataset_type = False, randomize_samples = False
                duplicate rows can appear in the same dataset type files
                the file will look the same after repeated calls to generate_dataset
                use this setting to generate files that overlap but deterministic
            
            ensure_uniqueness_across_same_dataset_type = False, randomize_samples = True
                duplicate rows can appear in the same dataset type files
                the file will not look the same after repeated calls to generate_dataset
                use this setting for complete randomness
            
            return value: list of tuples consisting of 
                the output file from the entry in the CSV config file,i.e., train_10K.csv or test_100K.csv
                the value for ensure_uniqueness_across_same_dataset_type,
                the value for randomize_samples,
                pd.DataFrame of the combined output for the output file.
        """

        ### Default to DataGenerator.GEN_DATA_INFO_FILE_NAME if info_file is empty/missing
        info_file = info_file or DataGenerator.GEN_DATA_INFO_FILE_NAME

        ### Read the metadata csv file
        try:
            info_df = pd.read_csv(info_file, 
                                  header=None, 
                                  names=[DataGenerator.GEN_DATA_INFO_DATA_TYPE_COLUMN_NAME, 
                                         DataGenerator.GEN_DATA_INFO_SAMPLE_SIZE_COLUMN_NAME, 
                                         DataGenerator.GEN_DATA_INFO_OUTPUT_FILE_COLUMN_NAME])
        except FileNotFoundError:
            print(f"Error: File '{info_file}' not found.")
            return []
        
        results = []
        
        # Process each row in metadata file, gen_data_info
        for _, row in info_df.iterrows():
            dataset_type = row[DataGenerator.GEN_DATA_INFO_DATA_TYPE_COLUMN_NAME]
            sample_size = row[DataGenerator.GEN_DATA_INFO_SAMPLE_SIZE_COLUMN_NAME]
            output_file = row[DataGenerator.GEN_DATA_INFO_OUTPUT_FILE_COLUMN_NAME].strip()
            
            # Validate dataset type
            if dataset_type not in self.dataset_map:
                print(f"Error: Invalid dataset type '{dataset_type}' for {output_file}. Skipping.")
                continue
            
            source_df = self.dataset_map[dataset_type]
            
            # Check label distribution
            label_counts = source_df[KaggleDataSet.POLARITY_COLUMN_NAME].value_counts()
            print(f"\nProcessing {output_file} from {dataset_type} dataset")
            print("Label counts:\n", label_counts)
            
            # Ensure enough rows for each label (50% split)
            half_size = sample_size // DataGenerator.HALF_SIZE
            
            # Filter available rows based on used indices if ensure_uniqueness_across_same_dataset_type is True
            if ensure_uniqueness_across_same_dataset_type:
                available_df = source_df[~source_df.index.isin(self.used_indices[dataset_type])]
                label_counts_available = available_df[KaggleDataSet.POLARITY_COLUMN_NAME].value_counts()
                if (label_counts_available.get(KaggleDataSet.POLARITY_VALUE_1, 0) < half_size or 
                    label_counts_available.get(KaggleDataSet.POLARITY_VALUE_2, 0) < half_size):
                    print(f"Error: Not enough unique rows for label {KaggleDataSet.POLARITY_VALUE_1} or "
                          f"label {KaggleDataSet.POLARITY_VALUE_2} to sample {half_size} each in {output_file}. Skipping.")
                    continue
            else:
                available_df = source_df
                if (label_counts.get(KaggleDataSet.POLARITY_VALUE_1, 0) < half_size or 
                    label_counts.get(KaggleDataSet.POLARITY_VALUE_2, 0) < half_size):
                    print(f"Error: Not enough rows for label {KaggleDataSet.POLARITY_VALUE_1} or "
                          f"label {KaggleDataSet.POLARITY_VALUE_2} to sample {half_size} each in {output_file}. Skipping.")
                    continue
            
            # Sample half_size rows for each label
            if randomize_samples:
                ### Shuffle the sampling of rows from each label
                sample_1 = available_df[available_df[KaggleDataSet.POLARITY_COLUMN_NAME] == KaggleDataSet.POLARITY_VALUE_1].sample(n=half_size)
                sample_2 = available_df[available_df[KaggleDataSet.POLARITY_COLUMN_NAME] == KaggleDataSet.POLARITY_VALUE_2].sample(n=half_size)
            else:
                ### Do not shuffle sample, use the same seed for each call to this method so the row "sampling" from each label is the same
                sample_1 = available_df[available_df[KaggleDataSet.POLARITY_COLUMN_NAME] == KaggleDataSet.POLARITY_VALUE_1].sample(
                    n=half_size, random_state=DataGenerator.RANDOMIZE_STATE_VAL)
                sample_2 = available_df[available_df[KaggleDataSet.POLARITY_COLUMN_NAME] == KaggleDataSet.POLARITY_VALUE_2].sample(
                    n=half_size, random_state=DataGenerator.RANDOMIZE_STATE_VAL)

            # Combine the samples
            downsampled_df = pd.concat([sample_1, sample_2])
            
            if randomize_samples:
                # Shuffle the combined DataFrame
                ####downsampled_df = downsampled_df.sample(frac=1, random_state=DataGenerator.RANDOMIZE_STATE_VAL).reset_index(drop=True)
                downsampled_df = downsampled_df.sample(frac=1).reset_index(drop=True)

            else:
                ### DO not shuffle the data frame, let the ordering of the pd.concat takes care of the ordering which should be deterministic
                downsampled_df = downsampled_df.reset_index(drop=True)
            
            # Save to CSV file
            downsampled_df.to_csv(output_file, index=False, header=False)
            
            # Update used indices if ensure_unique is True
            if ensure_uniqueness_across_same_dataset_type:
                self.used_indices[dataset_type].update(sample_1.index)
                self.used_indices[dataset_type].update(sample_2.index)
            
            # Verify the save
            print(f"Downsampled dataset saved to '{output_file}', ensure uniqueness across dataset type: {ensure_uniqueness_across_same_dataset_type}, randomize samples: {randomize_samples}")
            print("Downsampled dataset label counts:\n", downsampled_df[KaggleDataSet.POLARITY_COLUMN_NAME].value_counts())
            print("Downsampled dataset size:", downsampled_df.shape)
            
            ##
            ### Return the generated output file's name which comes from the CSV config file
            ###        ensure_uniqueness_across_same_dataset_type, randomize_samples,
            ###        and the combined pd.DataFrame which is writtent to output_file
            ###
            results.append((output_file, ensure_uniqueness_across_same_dataset_type, randomize_samples, downsampled_df))
        
        return results