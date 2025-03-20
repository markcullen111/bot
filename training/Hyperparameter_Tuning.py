# Hyperparameter_Tuning.py


import numpy as np
import json
import os
import optuna
import torch
import logging
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

class AutoHyperparameterTuner:
    """Adaptive hyperparameter tuning using Bayesian Optimization and Ray Tune"""
    
    def __init__(self, model_type='ml', max_trials=100, use_ray=True, storage_path='models/hyperparams'):
        self.model_type = model_type
        self.max_trials = max_trials
        self.use_ray = use_ray
        self.best_params = {}
        os.makedirs(storage_path, exist_ok=True)

    def tune_model(self, model_class, X_train, y_train, param_space, metric='neg_mean_squared_error'):
        """Tunes ML models using Optuna Bayesian Optimization"""
        def objective(trial):
            params = {k: self._suggest(trial, k, v) for k, v in param_space.items()}
            model = model_class(**params)
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                y_pred = model.predict(X_train.iloc[val_idx])
                scores.append(-mean_squared_error(y_train.iloc[val_idx], y_pred))
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.max_trials)
        self.best_params = study.best_params
        return self.best_params
    
    def tune_deep_learning(self, model_builder, X_train, y_train, param_space, n_epochs=50, batch_size=32):
        """Tunes deep learning models using Ray Tune"""
        def train_model(config):
            model = model_builder(**config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            loss_fn = torch.nn.MSELoss()
            for _ in range(n_epochs):
                optimizer.zero_grad()
                output = model(torch.FloatTensor(X_train))
                loss = loss_fn(output.squeeze(), torch.FloatTensor(y_train))
                loss.backward()
                optimizer.step()
            tune.report(loss=loss.item())
        
        scheduler = ASHAScheduler(max_t=n_epochs, grace_period=5, reduction_factor=2)
        analysis = tune.run(train_model, config=param_space, num_samples=self.max_trials, scheduler=scheduler)
        self.best_params = analysis.best_config
        return self.best_params
    
    def _suggest(self, trial, name, param):
        if param['type'] == 'float':
            return trial.suggest_float(name, param['min'], param['max'], log=param.get('log', False))
        elif param['type'] == 'int':
            return trial.suggest_int(name, param['min'], param['max'])
        elif param['type'] == 'categorical':
            return trial.suggest_categorical(name, param['values'])
    
    def save_results(self, strategy_name='default'):
        path = f'models/hyperparams/{strategy_name}.json'
        with open(path, 'w') as f:
            json.dump({'best_params': self.best_params}, f, indent=4)
    
    def load_results(self, strategy_name='default'):
        path = f'models/hyperparams/{strategy_name}.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.best_params = json.load(f)['best_params']
                return self.best_params
        return None

