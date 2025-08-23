import os 
from pathlib import Path
import pandas as pd

def export_dataset():
    main_root = Path(__file__).resolve().parents[1]
    dataset_path = main_root / 'data_cache' / 'cleanedFraudDataset.csv'

    df = pd.read_csv(dataset_path)
    df.to_csv('data.csv', index=False)
    
    
if __name__ == '__main__':
    export_dataset()