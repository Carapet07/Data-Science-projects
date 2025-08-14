import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)

class DataLoadingVisualizatin:
    def __init__(self, cache_path):
        self.df = None # It will store the loaded data
        self.cache_path = cache_path
        
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

            
        
class DataPreprocessing():
    pass

if __name__ == '__main__':
    data_path = "hf://datasets/dazzle-nu/CIS435-CreditCardFraudDetection/fraudTrain.csv"
        
    loader = DataLoadingVisualizatin(data_path)
    # loader.simple_data_showcase(data_path)

    loader.plot_numeric_distributions(data_path)
    loader.plot_class_balance(data_path)

    
    