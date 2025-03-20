# train_market_making_ai.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from market_making_ai import MarketMakingDQN

class MarketMakingTrainer:
    """Trains AI model for market-making strategies using reinforcement learning."""
    
    def __init__(self, model_path="models/market_making_model.pth", input_size=6, action_size=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarketMakingDQN(input_size, action_size).to(self.device)
        self.target_model = MarketMakingDQN(input_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.model_path = model_path
    
    def train_model(self, order_book_data, num_epochs=500):
        """Trains model using historical order book data and optimal bid/ask spreads."""
        for epoch in range(num_epochs):
            state, action, reward, next_state, done = self._sample_experience(order_book_data)
            self._train_step(state, action, reward, next_state, done)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Training... Loss: {self._calculate_loss():.4f}")
        
        torch.save(self.model.state_dict(), self.model_path)
        print("Market-Making AI Model Trained & Saved.")
    
    def _train_step(self, state, action, reward, next_state, done):
        """Performs a single training step using a replay memory sample."""
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)
        
        current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze()
        next_q = self.target_model(next_state).max(1)[0]
        expected_q = reward + (1 - done.float()) * self.gamma * next_q
        
        loss = self.criterion(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _calculate_loss(self):
        """Calculates the loss on a sample batch from memory."""
        if len(self.memory) < self.batch_size:
            return 0
        batch = np.random.choice(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q = self.model(states).gather(1, actions).squeeze()
        next_q = self.target_model(next_states).max(1)[0]
        expected_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        loss = self.criterion(current_q, expected_q.detach())
        return loss.item()
    
    def _sample_experience(self, order_book_data):
        """Samples a batch from the historical order book data."""
        idx = np.random.randint(0, len(order_book_data) - 1)
        state = order_book_data[idx, :-1]
        next_state = order_book_data[idx + 1, :-1]
        action = np.random.randint(0, 3)  # Buy, Sell, Hold
        reward = self._calculate_reward(state, action, next_state)
        done = idx == len(order_book_data) - 2
        return state, action, reward, next_state, done
    
    def _calculate_reward(self, state, action, next_state):
        """Reward function based on spread efficiency and profit."""
        bid_ask_spread = state[-1]
        price_change = next_state[-1] - state[-1]
        if action == 0:  # Buy
            return price_change - bid_ask_spread
        elif action == 1:  # Sell
            return -price_change - bid_ask_spread
        return 0  # Hold

# Load training data (replace with real dataset)
def load_training_data():
    df = pd.read_csv("data/historical_order_book.csv")
    return df.to_numpy()

if __name__ == "__main__":
    data = load_training_data()
    trainer = MarketMakingTrainer()
    trainer.train_model(data)

