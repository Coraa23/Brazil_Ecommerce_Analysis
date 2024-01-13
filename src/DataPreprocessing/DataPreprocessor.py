import pandas as pd

class DataProcessor:
    def __init__(self,data):
        """
        Initialize the DataProcessor with raw data.
        :param data: Pandas dataframe
        """
        self.raw_data = data
        self.processed_data = None

    def clean_data(self):
        """
        Data cleaning operation.
        :return: Pandas dataframe: cleaned data
        """
        # Drop rows with missing values
        cleaned_data = self.raw_data.dropna()
        # Drop rows with duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        self.processed_data = cleaned_data
        return cleaned_data

    def get_processed_data(self):
        """
        Return the processed data
        :return: Pandas dataframe: processed data
        """
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet.")
        return  self.processed_data