"""
Credit Card Fraud Detection - Main Training Pipeline
"""

from pathlib import Path
import pandas as pd
from data_preprocessing import DataLoadingVisualization, DataPreprocessing
from model_building import ModelTrainer

def data_preprocessing():
    """Load and preprocess the fraud detection dataset"""
    print("Starting data preprocessing...")
    
    # Load data
    data_path = "hf://datasets/dazzle-nu/CIS435-CreditCardFraudDetection/fraudTrain.csv"
    loader = DataLoadingVisualization(data_path)
    data = loader.load_or_read_dataset(data_path)
    print(f"Data loaded. Shape: {data.shape}")
    
    # Preprocess data
    preprocessor = DataPreprocessing(data)
    drop_columns = ['cc_num', 'trans_date_trans_time', 'trans_num', 'first', 'last', 'street', 'Unnamed: 0', '6006', 'Unnamed: 23']
    
    # Apply all preprocessing steps
    preprocessor.dropping_columns(drop_columns)    
    preprocessor.process_merchant_column()
    preprocessor.extract_ages()
    preprocessor.harvesine()
    preprocessor.convert_unitx_to_datetime()
    preprocessor.target_encodig()
    preprocessor.frequency_encoding()
    preprocessor.gender_encoding()
    preprocessor.amount_and_population_feature_engineering()
    
    # Save processed data
    project_dir = Path(__file__).resolve().parents[1]
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    preprocessor.df.to_csv(cleaned_data_path, index=False)
    print(f"Preprocessing completed. Saved to: {cleaned_data_path}")

def model_building():
    """Train all models and save the best one"""
    print("Starting model training...")
    
    # Load preprocessed data
    project_dir = Path(__file__).resolve().parents[1]
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    df = pd.read_csv(cleaned_data_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Train models
    trainer = ModelTrainer(dataframe=df, batch_size=32, epochs=20)
    best_model = trainer.save_best_model()
    print(f"Training completed. Best model: {best_model}")

def main():
    """Run the complete fraud detection pipeline"""
    print("="*50)
    print("CREDIT CARD FRAUD DETECTION PIPELINE")
    print("="*50)
    
    # Create directories
    project_dir = Path(__file__).resolve().parents[1]
    (project_dir / 'data_cache').mkdir(exist_ok=True)
    (project_dir / 'saved_models').mkdir(exist_ok=True)
    
    # Check if data is already processed
    cleaned_data_path = project_dir / 'data_cache' / 'cleanedFraudDataset.csv'
    
    if not cleaned_data_path.exists():
        data_preprocessing()
    else:
        print("Preprocessed data found. Skipping preprocessing.")
    
    # Train models
    model_building()
    print("="*50)
    print("PIPELINE COMPLETED!")
    print("="*50)

if __name__ == '__main__':
    main()