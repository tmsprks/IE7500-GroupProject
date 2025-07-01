
import sqlite3
import pandas as pd
import numpy as np
from kaggle_dataset import KaggleDataSet
from data_generator import DataGenerator

class SASQLiteDB:
    """A class to manage a SQLite database with separate tables for train, test, and validation datasets,
    including creation, updating, integrity checking, and data retrieval."""

    TITLE_DB_COLUMN_NAME = "title"
    REVIEW_DB_COLUMN_NAME = "review"
    LABEL_DB_COLUMN_NAME = "label"

    # Valid dataset types
    VALID_DATASET_TYPES = {DataGenerator.GEN_DATA_INFO_DATASET_TYPE_TRAIN_VAL, 
                           DataGenerator.GEN_DATA_INFO_DATASET_TYPE_TEST_VAL, 
                           DataGenerator.GEN_DATA_INFO_DATASET_TYPE_VALIDATION_VAL}

    def __init__(self, db_path):
        """
        Initialize the SQLiteDB with the path to the SQLite database file.
        
        Args:
            db_path (str): Path to the SQLite database file (e.g., 'reviews.db').
        """
        self.db_path = db_path

    def create_database(self, drop_if_exists=False):
        """
        Create the SQLite database with train_dataset, test_dataset, and validation_dataset tables.
        
        Args:
            drop_if_exists (bool): If True, drop existing tables before creating them.
        
        Returns:
            bool: True if the database and schema were created successfully, False otherwise.
        """

        train_dataset_table_name = "train_dataset"
        test_dataset_table_name = "test_dataset"
        validation_dataset_table_name = "validation_dataset"

        try:
            # Connect to the SQLite database (creates the file if it doesn't exist)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Drop tables if they exist and drop_if_exists is True
                if drop_if_exists:
                    cursor.execute(f"DROP TABLE IF EXISTS {train_dataset_table_name}")
                    cursor.execute(f"DROP TABLE IF EXISTS {test_dataset_table_name}")
                    cursor.execute(f"DROP TABLE IF EXISTS {validation_dataset_table_name}")

                # Create the train_dataset table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS train_dataset (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        "{SASQLiteDB.TITLE_DB_COLUMN_NAME}" TEXT,
                        "{SASQLiteDB.REVIEW_DB_COLUMN_NAME}" TEXT NOT NULL,
                        "{SASQLiteDB.LABEL_DB_COLUMN_NAME}" INTEGER NOT NULL,
                        CHECK (label IN (1, 2))
                    )
                """)

                # Create the test_dataset table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS test_dataset (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        "{SASQLiteDB.TITLE_DB_COLUMN_NAME}" TEXT,
                        "{SASQLiteDB.REVIEW_DB_COLUMN_NAME}" TEXT NOT NULL,
                        "{SASQLiteDB.LABEL_DB_COLUMN_NAME}" INTEGER NOT NULL,
                        CHECK (label IN (1, 2))
                    )
                """)

                # Create the validation_dataset table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS validation_dataset (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        "{SASQLiteDB.TITLE_DB_COLUMN_NAME}" TEXT,
                        "{SASQLiteDB.REVIEW_DB_COLUMN_NAME}" TEXT NOT NULL,
                        "{SASQLiteDB.LABEL_DB_COLUMN_NAME}" INTEGER NOT NULL,
                        CHECK (label IN (1, 2))
                    )
                """)

                # Commit the changes
                conn.commit()
                print(f"Database created successfully at '{self.db_path}' with train_dataset, test_dataset, and validation_dataset tables.")
                return True

        except sqlite3.Error as e:
            print(f"Error creating database: {e}")
            return False

    def update_database(self, dataset_type, dataframe):
        """
        Update the specified dataset table with rows from the provided DataFrame.
        
        Args:
            dataset_type (str): The type of dataset ('train', 'test', or 'validation').
            dataframe (pd.DataFrame): DataFrame with columns 'Title', 'Review', and 'Label'.
        
        Returns:
            bool: True if the database was updated successfully, False otherwise.
        """
        # Validate dataset_type
        if dataset_type not in self.VALID_DATASET_TYPES:
            print(f"Error: Invalid dataset_type '{dataset_type}'. Must be one of {self.VALID_DATASET_TYPES}.")
            return False

        # Validate DataFrame columns
        required_columns = {KaggleDataSet.TITLE_COLUMN_NAME, 
                            KaggleDataSet.REVIEW_COLUMN_NAME, 
                            KaggleDataSet.POLARITY_COLUMN_NAME}
        if not required_columns.issubset(dataframe.columns):
            missing = required_columns - set(dataframe.columns)
            print(f"Error: DataFrame is missing required columns: {missing}.")
            return False

        # Map dataset_type to table name
        table_name = f"{dataset_type}_dataset"

        try:
            # Connect to the SQLite database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Prepare the INSERT query
                insert_query = f"""
                    INSERT INTO {table_name} ({SASQLiteDB.TITLE_DB_COLUMN_NAME}, {SASQLiteDB.REVIEW_DB_COLUMN_NAME}, {SASQLiteDB.LABEL_DB_COLUMN_NAME})
                    VALUES (?, ?, ?)
                """.format(table_name)

                # Insert each row
                for _, row in dataframe.iterrows():
                    # Convert empty or NaN Title to None (NULL in SQLite)
                    title = None if pd.isna(row[KaggleDataSet.TITLE_COLUMN_NAME]) or row[KaggleDataSet.TITLE_COLUMN_NAME] == "" else row[KaggleDataSet.TITLE_COLUMN_NAME]
                    values = (title, row[KaggleDataSet.REVIEW_COLUMN_NAME], row[KaggleDataSet.POLARITY_COLUMN_NAME])
                    cursor.execute(insert_query, values)

                # Commit the changes
                conn.commit()
                print(f"Successfully updated '{table_name}' with {len(dataframe)} rows.")
                return True

        except sqlite3.Error as e:
            print(f"Error updating database: {e}")
            return False