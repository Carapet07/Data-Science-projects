from data_preprocessing import DataLoadingVisualization, DataPreprocessing
from pathlib import Path

def main():
    data_path = "hf://datasets/dazzle-nu/CIS435-CreditCardFraudDetection/fraudTrain.csv"
    loader = DataLoadingVisualization(data_path)
    data = loader.load_or_read_dataset(data_path)
    
    
    project_dir = Path(__file__).resolve().parents[1]
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    if cleaned_data_path.exists():
        Path.unlink(cleaned_data_path, missing_ok=True)  # delete the previous file before running a new experiment
    data.to_csv(cleaned_data_path, index=False)  # creates a new cleanedFraudDataset.csv file
        
        
    preprocessor = DataPreprocessing(data)  # takes the dataframe stored locally in the data_cache/fraudDataset.csv
    drop_columns = ['cc_num', 'trans_date_trans_time', 'trans_num', 'first', 'last', 'street', 'Unnamed: 0', '6006', 'Unnamed: 23']
    
    
    clean_data = preprocessor.dropping_columns(drop_columns)    
    clean_data = preprocessor.read_preprocessed_df()  
    clean_data = preprocessor.process_merchant_column()
    clean_data = preprocessor.extract_ages()
    clean_data = preprocessor.harvesine()
    clean_data = preprocessor.convert_unitx_to_datetime()
    clean_data = preprocessor.target_encodig()
    clean_data = preprocessor.frequency_encoding()
    clean_data = preprocessor.gender_encoding()
    clean_data = preprocessor.amount_and_population_feature_engineering()

    clean_data.to_csv(cleaned_data_path, index=False)
    print('The preprocessing step is COMPLETED!')


if __name__ == '__main__':
    main()
    
    