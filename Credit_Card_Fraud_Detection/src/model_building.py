import numpy as np
from pathlib import Path

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# custom models
from xgboost import XGBClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import SVC




class DataPreparation():
    def __init__(self, dataframe, batch_size):
        self.df = dataframe
        self.batch_size = batch_size
    
        
    def get_data_dir(self):
        # Get the dataset path
        project_root = Path(__file__).resolve().parents(1)
        return project_root / 'data_cache' / 'cleanedFraudDataset.csv'
    
    
    def train_test_val_splits(self):
        X = self.df.drop('is_fraud', axis=1).values.astype(np.float32)
        y = self.df['is_fraud'].values.astype(np.int64)
        
        
        train_X, temp_X, train_y, temp_y = train_test_split(
            X, y,
            train_size=0.7, 
            random_state=42, 
            stratify=y
        )
        
        test_X, val_X, test_y, val_y = train_test_split(
            temp_X, temp_y,
            train_size=0.5,
            random_state=42,
            stratify=temp_y
        )
        
        X_train = torch.FloatTensor(train_X)
        y_train = torch.FloatTensor(train_y) 
        X_val = torch.FloatTensor(val_X)
        y_val = torch.FloatTensor(val_y) 
        X_test = torch.FloatTensor(test_X)
        y_test = torch.FloatTensor(test_y)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
       )
        
        return train_loader, val_loader, test_loader


    def to_numpy(self):
        """
        Pretrained models such as xbs, svm, random forest don't support directly the pytorch's
        data loaders, that is why we have to convert them back to numpy arrays for those models 
        """
        X = self.df.drop('is_fraud', axis=1)
        y = self.df['is_fraud']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            train_size=0.7,
            random_state=42,
            stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            train_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    
class MLPBinary(nn.Module):
    def __init__(self, input_dim):
        """
        Binary classification neural network
        
        Args:
            input_dim (int): Number of input features
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.linear2 = nn.Linear(128, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.linear3 = nn.Linear(64, 32)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        
        return x
    
    
class ModelTrainer:
    def __init__(self, dataframe, batch_size, epochs):
        self.dataframe = dataframe
        self.epochs = epochs
        
        # Setting up the device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create data preparation instance
        data_prep = DataPreparation(dataframe, batch_size)
        
        # Get PyTorch DataLoaders for neural networks
        self.train_loader, self.val_loader, self.test_loader = data_prep.train_test_val_splits()
        
        # Get numpy arrays for scikit-learn models 
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = data_prep.to_numpy()
        print('Data Prepared')

    def mlp_train(self):
        input_dim = len(self.dataframe.columns) - 1  # -1 for 'is_fraud' column
        logist_model = MLPBinary(input_dim)
        logist_model = logist_model.to(self.device)
        
        optimizer = torch.optim.Adam(logist_model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()


        steps_per_epoch = len(self.train_loader)
        logist_model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (inputs, label) in self.train_loader:
                inputs = inputs.to(self.device) 
                label = label.to(self.device)
                
                outputs = logist_model(inputs)
                loss = criterion(outputs.view(-1), label)
                  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"MLP model: [{epoch+1}]: loss: {(total_loss / steps_per_epoch):.3f}")
            
        return logist_model


    def train_xgb(self):
        xgb = XGBClassifier(
            n_estimators=800,  # This is like "epochs" for XGBoost
            max_depth=8, 
            learning_rate=0.05,
            subsample=0.9, 
            colsample_bytree=0.9,
            objective="binary:logistic", 
            eval_metric="logloss",
            tree_method="hist", 
            random_state=0
        )
        
        xgb.fit(self.X_train, self.y_train, 
                eval_set=[(self.X_val, self.y_val)], 
                verbose=0)
        return xgb
    
    
    def train_rand_forest(self):
        rand_forest = RandomForestClassifier(
            n_estimators=500,  # Number of trees (like "iterations")
            max_depth=None,
            random_state=0,  # Fixed typo
            n_jobs=-1,  # Use all CPU cores
            class_weight="balanced_subsample"
        )                
        rand_forest.fit(self.X_train, self.y_train)  # No batch_size/epochs for sklearn
        return rand_forest
    
    
    def train_svm_brf(self):
        # SVM needs scaling
        scaler = StandardScaler().fit(self.X_train)
        Xtr = scaler.transform(self.X_train)
        Xv = scaler.transform(self.X_val)
        
        svm = SVC(
            kernel="rbf", 
            C=1.0, 
            gamma="scale", 
            probability=True, 
            class_weight="balanced", 
            random_state=0
        )
        svm.fit(Xtr, self.y_train)  # No batch_size/epochs for SVM
        return (svm, scaler)
    


    def evaluate_torch(self):
        """
        This function is for evaluating the Pytorch's MLP model I've made above
        """
        self.logist_model.eval()
        
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for (inputs, label) in self.test_loader:
                inputs = inputs.to(self.device) 
                label = label.to(self.device)
                
                logits = self.logist_model(inputs)
                predictions = torch.sigmoid(logits) > 0.5
                
                total_correct += (predictions.view(-1) == label).sum().item()
                total_samples += label.size(0)
                
                all_predictions.extend(predictions.view(-1).cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        
        accuracy = total_correct / total_samples
        print(f'Test Accuracy: {accuracy:.4f}')
        
        return accuracy, all_predictions, all_labels


    def evaluate_sklearn(self, model, scaler=None):
        """
        Evaluate scikit-learn models (XGBoost, RandomForest, SVM)
        
        Args:
            model: Trained sklearn/xgboost model
            scaler: Optional scaler (needed for SVM)
        
        Returns:
            accuracy, predictions, true_labels
        """

        X_test = self.X_test
        y_test = self.y_test
         
        # Apply scaling if provided (for SVM)
        if scaler is not None:
            X_test = scaler.transform(X_test)
        
        # Make predictions
        predictions = model.predict(X_test)
        prediction_probs = model.predict_proba(X_test)[:, 1]  # Get fraud probabilities
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print(f'Test Accuracy: {accuracy:.4f}')
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Not Fraud', 'Fraud']))
        
        return accuracy, predictions, y_test, prediction_probs
    
    def evaluate_all_models(self):
        """
        Train and evaluate all models for comparison
        """
        print("=" * 50)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("=" * 50)
        
        results = {}
        
        # 1. PyTorch MLP
        print("\n1. Training PyTorch MLP...")
        mlp_model = self.mlp_train()
        mlp_accuracy, _, _ = self.evaluate_torch()
        results['MLP'] = mlp_accuracy
        
        # 2. XGBoost
        print("\n2. Training XGBoost...")
        xgb_model = self.train_xgb()
        xgb_accuracy, _, _, _ = self.evaluate_sklearn(xgb_model)
        results['XGBoost'] = xgb_accuracy
        
        # 3. Random Forest
        print("\n3. Training Random Forest...")
        rf_model = self.train_rand_forest()
        rf_accuracy, _, _, _ = self.evaluate_sklearn(rf_model)
        results['Random Forest'] = rf_accuracy
        
        # 4. SVM RBF
        print("\n4. Training SVM RBF...")
        svm_model, scaler = self.train_svm_rbf()
        svm_accuracy, _, _, _ = self.evaluate_sklearn(svm_model, scaler)
        results['SVM RBF'] = svm_accuracy
        
        # Summary
        print("\n" + "=" * 50)
        print("MODEL COMPARISON RESULTS")
        print("=" * 50)
        
        for model, prediction in results.items():
            print(f"{model:>15} {prediction:.4f} ({prediction*100:.4f})")
            
        best_model = max(results, key=results.get)
        print(f"The {best_model} has the best accuracy of {results[best_model]:.4f}")
        return results