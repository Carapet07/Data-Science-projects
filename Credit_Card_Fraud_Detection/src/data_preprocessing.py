import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)

class DataLoadingVisualization:
    def __init__(self, cache_path):
        self.df = None # It will store the loaded data
        self.cache_path = None 
        
    def create_cache(self):
        proj_directory = Path(__file__).resolve().parents[1]
        cache_dir = proj_directory / 'data_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / 'fraudDataset.csv'
            
            
    def load_or_read_dataset(self, data):
        cache_path = self.create_cache()
        if cache_path.exists() and cache_path.stat().st_size > 0:
            self.df = pd.read_csv(cache_path)
            return self.df
        
        df = pd.read_csv(data)
        df.to_csv(cache_path, index=False)
        self.df = df
        return df
            
    def simple_data_showcase(self, dataset):
        df = self.load_or_read_dataset(dataset)
        print(f'Dataset"s columns: \n{df.columns}')
        print(f'Number of instances \n{len(df)}')
    
    
    def plot_numeric_distributions(self, data):
        df = self.load_or_read_dataset(data)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if numeric_cols:
            df[numeric_cols].hist(bins=30, figsize=(14, 10))
            plt.tight_layout()
            plt.show()
        else:
            print('No numeric columns found.')
        return df

    def plot_class_balance(self, data):
        df = self.load_or_read_dataset(data)
    
        label_candidates = [c for c in df.columns if c.lower() in {'is_fraud'}]
        if label_candidates:
            label_col = label_candidates[0]
            counts = df[label_col].value_counts()
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette='viridis')
            plt.title(f'Class Balance: {label_col}')
            plt.xlabel(label_col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
        else:
            print('No obvious label column found for class balance plot.')

            
            
            
        
class DataPreprocessing:
    """
    There is a list of columns we've got in the dataset:
    Index(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
       'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
       'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
       'merch_lat', 'merch_long', 'is_fraud', 'Unnamed: 23', '6006'],
      dtype='object')
    """
    def __init__(self, dataframe):
        self.original_df = dataframe  # creatae a reference to the original dataframe 
        self.df = dataframe.copy(deep=True)
        new_df = self.df
        
    def dropping_columns(self, columns_to_drop: list[str]) -> pd.DataFrame:
        df = self.df.drop(columns=columns_to_drop, errors='ignore')
        self.df = df
        return self.df
    
    def process_merchant_column(self) -> pd.DataFrame:
        le = LabelEncoder()
        if 'merchant' not in self.df.columns:
            print('Merchant columns is not found')
            return self.df
        self.df['merchant_encoding'] = le.fit_transform(self.df['merchant'])
        self.df = self.df.drop('merchant', axis=1)
        print('Merchant columns was encoded')
        return self.df
    
    def extract_ages(self) -> pd.DataFrame:
        """
        Convert date of birth to age, handling century cutoff issue
        Pandas uses a century cutoff where years 00-69 are interpreted as
        2000-2069, and years 70-99 are interpreted as 1970-1999.
        """
        if 'dob' not in self.df.columns:
            print('DOB not found')
            return self.df
        
        birthday = pd.to_datetime(self.df['dob'], format='%m/%d/%y')
        birthday_year = birthday.dt.year
        
        # To handle the panda's century problem we keep years which are below 2000 
        # and those above this value are subtracting by 100
        corrected_years = birthday_year.where(birthday_year < 2000, birthday_year-100)
        
        # Reference year based on transaction data
        current_year = 2019
        self.df['age'] = current_year - corrected_years
        self.df = self.df.drop('dob', axis=1)
        
        
        return self.df

        

    def read_preprocessed_df(self):
        print(self.df.columns)
        return self.df.head()


if __name__ == '__main__':
    data_path = "hf://datasets/dazzle-nu/CIS435-CreditCardFraudDetection/fraudTrain.csv"
        
    loader = DataLoadingVisualization(data_path)
    # loader.simple_data_showcase(data_path)

    loader.plot_numeric_distributions(data_path)
    loader.plot_class_balance(data_path)

    
    