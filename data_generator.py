
import pandas as pd

class DataGenerator:
    POLARITY_COLUMN_NAME = "Label"
    TITLE_COLUMN_NAME = "Title"
    REVIEW_COLUMN_NAME = "Review"
    POLARITY_VALUE_1 = 1
    POLARITY_VALUE_2 = 2
    HALF_SIZE = 2

    GEN_DATA_INFO_DATA_TYPE_COLUMN_NAME = "Data_Type"
    GEN_DATA_INFO_SAMPLE_SIZE_COLUMN_NAME = "Sample_Size"
    GEN_DATA_INFO_OUTPUT_FILE_COLUMN_NAME = "Output_File"

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, validation_df: pd.DataFrame):

        ### Maps data set type to the dataframe variable
        self.dataset_map = {
            'train': train_df,
            'test': test_df,
            'validation': validation_df
        }

        ### Initialize a dictionary to track used indices for each dataset type
        ### in order to ensure uniqueness
        self.used_indices = {
            'train': set(),
            'test': set(),
            'validation': set()
        }

    def generate_datasets(self, info_file: str='gen_data_info.csv', ensure_uniqueness: bool=False):
        # Read the metadata csv file
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
            label_counts = source_df[DataGenerator.POLARITY_COLUMN_NAME].value_counts()
            print(f"\nProcessing {output_file} from {dataset_type} dataset")
            print("Label counts:\n", label_counts)
            
            # Ensure enough rows for each label (50% split)
            half_size = sample_size // DataGenerator.HALF_SIZE
            
            # Filter available rows based on used indices if ensure_unique is True
            if ensure_uniqueness:
                available_df = source_df[~source_df.index.isin(self.used_indices[dataset_type])]
                label_counts_available = available_df[DataGenerator.POLARITY_COLUMN_NAME].value_counts()
                if (label_counts_available.get(DataGenerator.POLARITY_VALUE_1, 0) < half_size or 
                    label_counts_available.get(DataGenerator.POLARITY_VALUE_2, 0) < half_size):
                    print(f"Error: Not enough unique rows for label {DataGenerator.POLARITY_VALUE_1} or "
                          f"label {DataGenerator.POLARITY_VALUE_2} to sample {half_size} each in {output_file}. Skipping.")
                    continue
            else:
                available_df = source_df
                if (label_counts.get(DataGenerator.POLARITY_VALUE_1, 0) < half_size or 
                    label_counts.get(DataGenerator.POLARITY_VALUE_2, 0) < half_size):
                    print(f"Error: Not enough rows for label {DataGenerator.POLARITY_VALUE_1} or "
                          f"label {DataGenerator.POLARITY_VALUE_2} to sample {half_size} each in {output_file}. Skipping.")
                    continue
            
            # Sample half_size rows for each label
            sample_1 = available_df[available_df[DataGenerator.POLARITY_COLUMN_NAME] == DataGenerator.POLARITY_VALUE_1].sample(
                n=half_size, random_state=42)
            sample_2 = available_df[available_df[DataGenerator.POLARITY_COLUMN_NAME] == DataGenerator.POLARITY_VALUE_2].sample(
                n=half_size, random_state=42)
            
            # Combine the samples
            downsampled_df = pd.concat([sample_1, sample_2])
            
            # Shuffle the combined DataFrame
            downsampled_df = downsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save to CSV file
            downsampled_df.to_csv(output_file, index=False)
            
            # Update used indices if ensure_unique is True
            if ensure_uniqueness:
                self.used_indices[dataset_type].update(sample_1.index)
                self.used_indices[dataset_type].update(sample_2.index)
            
            # Verify the save
            print(f"Downsampled dataset saved to '{output_file}'")
            print("Downsampled dataset label counts:\n", downsampled_df[DataGenerator.POLARITY_COLUMN_NAME].value_counts())
            print("Downsampled dataset size:", downsampled_df.shape)
            
            # Store result
            results.append((output_file, downsampled_df))
        
        return results