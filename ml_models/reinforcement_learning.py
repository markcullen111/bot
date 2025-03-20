# reinforcement_learning.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class TradingEnvironment:
    """Enhanced trading environment for reinforcement learning"""
    
    def __init__(self, data, initial_balance=10000, window_size=30, commission=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.total_reward = 0
        self.done = False
        return self._get_state()
    
    def step(self, action):
        """Execute action and return next state, reward, done flag"""
        reward = self._calculate_reward(action)
        self.total_reward += reward
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        return self._get_state(), reward, self.done, {}
    
    def _get_state(self):
        """Get the current state representation"""
        return self.data[self.current_step - self.window_size:self.current_step]
    
    def _calculate_reward(self, action):
        """Improved reward function for RL trading"""
        price_change = self.data[self.current_step, 3] - self.data[self.current_step - 1, 3]
        
        if action == 1:  # Buy
            return price_change - self.commission
        elif action == 2:  # Sell
            return -price_change - self.commission
        return 0  # Hold

class DQNTrader(nn.Module):
    """Deep Q-Network (DQN) for reinforcement learning-based trading"""
    
    def __init__(self, input_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values

class StrategyOptimizationRL:
    """Reinforcement Learning System for Strategy Optimization"""
    
    def __init__(self, input_size=30, action_size=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNTrader(input_size, action_size).to(self.device)
        self.target_model = DQNTrader(input_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def train(self, batch_size=32):
        """Train the agent using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def predict_action(self, state):
        """Select an action using an epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experiences for training"""
        self.memory.append((state, action, reward, next_state, done))
        
    def update_target_network(self):
        """Periodically update the target network"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def adjust_exploration(self):
        """Gradually reduce epsilon for better learning"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

