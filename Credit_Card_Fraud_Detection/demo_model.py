"""
Demo model for Streamlit app when no trained model is available
This creates a simple mock model for demonstration purposes
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_demo_model():
    """Create a simple demo RandomForestClassifier model for the Streamlit app"""
    
    # Check if we have processed data
    data_path = Path("data_cache/cleanedFraudDataset.csv")
    
    if data_path.exists():
        print("Loading real data for demo model...")
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train a simple Random Forest
        model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Demo model trained!")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
    
    else: 
        print('No data path found')
        
    # Save the model
    models_dir = Path("saved_models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "best_model.pkl"
    joblib.dump(model, model_path)
    
    print(f"Demo model saved to: {model_path}")
    return model

if __name__ == "__main__":
    create_demo_model()


