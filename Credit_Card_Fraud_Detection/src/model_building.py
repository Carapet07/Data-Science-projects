import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



class DataPreparation():
    def __init__(self, dataframe, batch_size):
        self.df = dataframe
        self.batch_size = batch_size
    
        
    def get_data_dir(self):
        # Get the dataset path
        project_root = Path(__file__).resolve().parents(1)
        return project_root / 'data_cache' / 'cleanedFraudDataset.csv'
    
    
    def train_test_val_splits(self):
        X = self.df.drop('is_fraud', axis=1)
        y = self.df['is_fraud']
        
        
        train_X, temp_X, train_y, temp_y = train_test_split(
            X, y,
            train_size=70, 
            random_state=42, 
            stratify=y
        )
        
        test_X, val_X, test_y, val_y = train_test_split(
            temp_X, temp_y,
            train_size=50,
            random_state=42,
            stratify=temp_y
        )
        
        X_train = torch.FloatTensor(train_X.values)
        y_train = torch.FloatTensor(train_y.values)  # For BCELoss
        X_val = torch.FloatTensor(val_X.values)
        y_val = torch.FloatTensor(val_y.values)
        X_test = torch.FloatTensor(test_X.values)
        y_test = torch.FloatTensor(test_y.values)
        
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
    
    
class ModelBuilding(nn.Module):
    def __init__(self, input_dim):
        """
        Binary classification neural network
        
        Args:
            input_dim (int): Number of input features
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.linear2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.linear3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x
    
    
class ModelTrainer:
    def __init__(self, dataframe, batch_size, epochs):
        self.dataframe = dataframe
        self.epochs = epochs
        
        # Setting up the device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Create data loaders
        data_prep = DataPreparation(dataframe, batch_size)
        self.train_loader, self.val_loader, self.test_loader = data_prep.train_test_val_splits()
        
        # Get input dimension from dataframe (excluding target column)
        input_dim = len(dataframe.columns) - 1  # -1 for 'is_fraud' column
        
        # Initialize model with correct input dimension
        self.logist_model = ModelBuilding(input_dim)
        self.logist_model = self.logist_model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.logist_model.parameters(), lr=0.01, weight_decay=1e-4)
        self.criterion = nn.BCELoss()  # Use BCELoss since model has sigmoid
        
        
    def train(self):
        steps_per_epoch = len(self.train_loader)
        
        self.logist_model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (inputs, label) in self.train_loader:
                inputs = inputs.to(self.device) 
                label = label.to(self.device)
                
                outputs = self.logist_model(inputs)
                loss = self.criterion(outputs, label.unsqueeze(1))
                  
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"[{epoch+1}]: loss: {(total_loss / steps_per_epoch):.3f}")
            
        print('Training Finished!')
