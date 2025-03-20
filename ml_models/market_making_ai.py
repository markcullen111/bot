# market_making_ai.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class MarketMakingDQN(nn.Module):
    """Deep Q-Network (DQN) for AI-driven market-making"""
    def __init__(self, input_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class AIMarketMaker:
    """Reinforcement learning-based market-making agent"""
    def __init__(self, input_size=6, action_size=3, model_path="models/market_making_model.pth"):
        self.model = MarketMakingDQN(input_size, action_size)
        self.target_model = MarketMakingDQN(input_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0  # Exploration rate
        self.gamma = 0.99  # Discount factor
        self.replay_buffer = []
        self.batch_size = 32
        
    def predict_action(self, market_data):
        """Predicts bid/ask adjustments using the trained DQN"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action (exploration)
        
        features = torch.FloatTensor(market_data).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(features)
        return torch.argmax(action_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experiences for training"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)
        
    def train(self):
        """Trains the DQN using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        """Updates target network weights"""
        self.target_model.load_state_dict(self.model.state_dict())

