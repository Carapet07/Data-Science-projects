from data_preprocessing import DataLoadingVisualization, DataPreprocessing
from model_building import DataPreparation, MLPBinary, ModelTrainer
from pathlib import Path
import pandas as pd

def data_preprocessing():
    data_path = "hf://datasets/dazzle-nu/CIS435-CreditCardFraudDetection/fraudTrain.csv"
    loader = DataLoadingVisualization(data_path)
    data = loader.load_or_read_dataset(data_path)
    
    
    project_dir = Path(__file__).resolve().parents[1]
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    # Note: File existence check is now handled in main()
    data.to_csv(cleaned_data_path, index=False)  # creates a new cleanedFraudDataset.csv file
        
        
    preprocessor = DataPreprocessing(data)  # takes the dataframe stored locally in the data_cache/fraudDataset.csv
    drop_columns = ['cc_num', 'trans_date_trans_time', 'trans_num', 'first', 'last', 'street', 'Unnamed: 0', '6006', 'Unnamed: 23']
    
    
    # Chain preprocessing functions properly
    preprocessor.dropping_columns(drop_columns)    
    preprocessor.process_merchant_column()
    preprocessor.extract_ages()
    preprocessor.harvesine()
    preprocessor.convert_unitx_to_datetime()
    preprocessor.target_encodig()
    preprocessor.frequency_encoding()
    preprocessor.gender_encoding()
    preprocessor.amount_and_population_feature_engineering()
    
    # Get the final processed DataFrame
    clean_data = preprocessor.df
    clean_data.to_csv(cleaned_data_path, index=False)
    print('The preprocessing step is COMPLETED!')
    
    

def model_building():
    project_dir = Path(__file__).resolve().parents[1]
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    df = pd.read_csv(cleaned_data_path)

    
    trainer = ModelTrainer(dataframe=df,
                           batch_size=32,
                           epochs=1)
    results = trainer.mlp_train()
    evaluation_metrics = trainer.evaluate_torch()
    #print(evaluation_metrics)
    
    
if __name__ == '__main__':
    # Check if cleaned data already exists
    project_dir = Path(__file__).resolve().parents[1]
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    
    if not cleaned_data_path.exists():
        print("Cleaned data not found. Running preprocessing...")
        data_preprocessing()
    else:
        print("Cleaned data found. Skipping preprocessing...")
    
    model_building()
    
    
    