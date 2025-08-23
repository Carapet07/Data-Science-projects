import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder


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

        
    def harvesine(self) -> pd.DataFrame:
        """
        This function calculate distance between user and merchant
        The haversine formula is a method for calculating the great-circle distance between 
        two points on a sphere (like Earth) given their latitude and longitude coordinates
        """
        required_cols = ['lat', 'long', 'merch_lat', 
                         'merch_long']
        if not all(col in self.df.columns for col in required_cols):
            print('Missing expected columns')
            return self.df
        
        
        R = 6371  # Earth radius in km
        phi1 = np.radians(self.df['lat'])
        phi2 = np.radians(self.df['merch_lat'])
        dphi = np.radians(self.df['merch_lat'] - self.df['lat'])
        dlambda = np.radians(self.df['merch_long'] - self.df['long'])

        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        result = R * c
        self.df['user_merchant_distance_km'] = result.round(1)
        self.df = self.df.drop(required_cols, axis=1)
        return self.df
    

    def convert_unitx_to_datetime(self):
        self.df['time'] = pd.to_datetime(self.df['unix_time'], unit='s')
        
        self.df['day_of_week'] = self.df['time'].dt.dayofweek
        self.df['hour'] = self.df['time'].dt.hour
        self.df['month'] = self.df['time'].dt.month
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)        
        self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 5)).astype(int)
        
        self.df = self.df.drop(['unix_time', 'time'], axis=1)
        return self.df


    def target_encodig(self):
        self.df['category_encoded'] = self.df['category'].map(
            self.df.groupby('category')['is_fraud'].mean()
        )
        self.df['state_encoded'] = self.df['state'].map(
            self.df.groupby('state')['is_fraud'].mean()
        )
        self.df['zip_encoded'] = self.df['zip'].map(
            self.df.groupby('zip')['is_fraud'].mean()
        )
        
        self.df = self.df.drop(['category', 'state', 'zip'], axis=1)
        return self.df

        
    def frequency_encoding(self):
        job_freq = self.df['job'].value_counts(normalize=True)
        self.df['job_freq'] = self.df['job'].map(job_freq)
        
        city_freq = self.df['city'].value_counts(normalize=True)
        self.df['city_freq'] = self.df['city'].map(city_freq)
        
        self.df = self.df.drop(['job', 'city'], axis=1)
        return self.df


    def gender_encoding(self):
        self.df['gender_encoded'] = (self.df['gender'] == "M").astype(int)
        self.df = self.df.drop('gender', axis=1)
        return self.df

    def amount_and_population_feature_engineering(self):
        self.df['amt_log'] = np.log1p(self.df['amt'])
        self.df['city_pop_log'] = np.log1p(self.df['city_pop'])

        
        self.df['amt_bin'] = pd.cut(self.df['amt_log'],
                                    bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        self.df['city_pop_bin'] = pd.cut(self.df['city_pop'], 
                                        bins=5, labels=['very_small', 'small', 'medium', 'large', 'very_large'])
        
        # Convert categorical bins to numeric (ordinal encoding)
        amt_mapping = {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        pop_mapping = {'very_small': 0, 'small': 1, 'medium': 2, 'large': 3, 'very_large': 4}
        
        self.df['amt_bin_encoded'] = self.df['amt_bin'].map(amt_mapping)
        self.df['city_pop_bin_encoded'] = self.df['city_pop_bin'].map(pop_mapping)
        
        # Drop the original categorical bins
        self.df = self.df.drop(['amt_bin', 'city_pop_bin'], axis=1)
        
        # Check for outliers
        outlier_columns = ['amt', 'city_pop']
        for col in outlier_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df[f'{col}_outlier'] = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                                        (self.df[col] > (Q3 + 1.5 * IQR))).astype(int)

        self.df = self.df.drop(['amt', 'city_pop'], axis=1)
        return self.df
    
    
    
    
    
    
    def read_preprocessed_df(self):
        print(self.df.columns)
        return self.df.head()
    
    