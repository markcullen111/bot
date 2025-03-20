# train_model.py


import os
import logging
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor

# Try importing CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logging.warning("CatBoost not available, skipping this model")

logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    """Trains and evaluates ML models for market prediction."""
    
    def __init__(self, save_path="models"):
        self.save_path = save_path
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        os.makedirs(save_path, exist_ok=True)
    
    def prepare_data(self, data, target_column='close', features=None):
        """Prepares market data for training."""
        df = data.copy()
        target = df[target_column].shift(-1)
        
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features.remove(target_column)
        
        X = df[features]
        y = target.dropna()
        X = X.loc[y.index]  # Align features with target
        logging.info(f"Prepared {len(X)} samples with {len(features)} features")
        return X, y
    
    def train_and_evaluate(self, X, y, test_size=0.2, random_state=42):
        """Trains multiple models and evaluates performance."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        models = {
            'rf': RandomForestRegressor(n_estimators=200, random_state=random_state),
            'xgb': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state),
            'lgbm': LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state)
        }
        
        if HAS_CATBOOST:
            models['catboost'] = CatBoostRegressor(iterations=200, learning_rate=0.05, random_seed=random_state, verbose=False)
        
        for name, model in models.items():
            logging.info(f"Training {name}...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            self.models[name] = model
            self.scalers[name] = scaler
            self.metrics[name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            logging.info(f"{name} RMSE: {self.metrics[name]['rmse']:.4f}, R²: {self.metrics[name]['r2']:.4f}")
    
    def train_neural_network(self, X, y, epochs=50, batch_size=32):
        """Trains a neural network for market prediction."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train_scaled, X_test_scaled = scaler_X.fit_transform(X_train), scaler_X.transform(X_test)
        y_train_scaled, y_test_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(), scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        class PricePredictor(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, 128), nn.Linear(128, 64), nn.Linear(64, 1)
                self.relu, self.dropout = nn.ReLU(), nn.Dropout(0.2)
            def forward(self, x):
                x = self.relu(self.fc1(x)); x = self.dropout(x)
                x = self.relu(self.fc2(x)); return self.fc3(x)
        
        device, input_dim = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), X_train_scaled.shape[1]
        model, optimizer, criterion = PricePredictor(input_dim).to(device), optim.Adam(model.parameters(), lr=0.001), nn.MSELoss()
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_train_scaled).to(device), torch.FloatTensor(y_train_scaled).to(device)), batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs).squeeze(), targets)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        self.models['nn'], self.scalers['nn_X'], self.scalers['nn_y'] = model, scaler_X, scaler_y
        logging.info("Neural network training complete.")
    
    def save_models(self):
        """Saves trained models."""
        save_dir = os.path.join(self.save_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            if name == 'nn': torch.save(model.state_dict(), os.path.join(save_dir, "nn_model.pth"))
            else: joblib.dump(model, os.path.join(save_dir, f"{name}_model.pkl")); joblib.dump(self.scalers[name], os.path.join(save_dir, f"{name}_scaler.pkl"))
        logging.info(f"Models saved to {save_dir}")
    
    def predict(self, X):
        """Predicts using an ensemble of trained models."""
        if not self.models: logging.error("No trained models available."); return None
        X_scaled, predictions = self.scalers['rf'].transform(X), []
        for name, model in self.models.items(): predictions.append(model.predict(X_scaled) if name != 'nn' else self.scalers['nn_y'].inverse_transform(model(torch.FloatTensor(self.scalers['nn_X'].transform(X)).to('cuda' if torch.cuda.is_available() else 'cpu')).detach().cpu().numpy().reshape(-1, 1)).flatten())
        return np.mean(predictions, axis=0)

if __name__ == "__main__":
    trainer = ModelTrainer()
    data = pd.read_csv("data/market_data.csv")
    X, y = trainer.prepare_data(data)
    trainer.train_and_evaluate(X, y)
    trainer.train_neural_network(X, y)
    trainer.save_models()

