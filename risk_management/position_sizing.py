# Adaptive Position Sizing with Risk Management

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
import math

class AdaptivePositionSizer:
    """
    Adaptive position sizing system with dynamic risk management
    Adjusts position sizes based on confidence, volatility, and performance
    """
    
    def __init__(self, base_risk=0.02, max_risk=0.05, min_risk=0.01, initial_capital=10000, 
                 save_path='models/position_sizing'):
        """
        Initialize the adaptive position sizer
        
        Args:
            base_risk: Base risk per trade (percentage of portfolio)
            max_risk: Maximum risk per trade
            min_risk: Minimum risk per trade
            initial_capital: Initial capital
            save_path: Path to save position sizing data
        """
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.save_path = save_path
        
        # Performance tracking
        self.trades = []
        self.win_streak = 0
        self.loss_streak = 0
        self.recent_trades_pnl = []  # Track recent trade PnLs
        self.max_drawdown = 0
        
        # Dynamic risk settings
        self.volatility_factor = 1.0
        self.confidence_factor = 1.0
        self.streak_factor = 1.0
        self.equity_curve_factor = 1.0
        
        # Market condition risk adjustments
        self.market_condition_factors = {
            'bull': 1.2,    # More aggressive in bull markets
            'bear': 0.8,    # More conservative in bear markets
            'sideways': 0.9  # Slightly conservative in sideways markets
        }
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def calculate_position_size(self, price, confidence=1.0, volatility=1.0, 
                               market_condition=None, prediction_type=None):
        """
        Calculate optimal position size based on current conditions
        
        Args:
            price: Current asset price
            confidence: Model confidence (0-1)
            volatility: Market volatility measure (1.0 = normal)
            market_condition: Current market condition (e.g., 'bull', 'bear')
            prediction_type: Type of prediction ('trend', 'reversal', etc.)
            
        Returns:
            Dictionary with position size information
        """
        # Update factors
        self.confidence_factor = self._calculate_confidence_factor(confidence)
        self.volatility_factor = self._calculate_volatility_factor(volatility)
        self.streak_factor = self._calculate_streak_factor()
        self.equity_curve_factor = self._calculate_equity_curve_factor()
        
        # Get market condition factor
        market_factor = self.market_condition_factors.get(market_condition, 1.0) if market_condition else 1.0
        
        # Get prediction type factor
        prediction_factor = self._get_prediction_type_factor(prediction_type)
        
        # Calculate dynamic risk
        risk = self.base_risk * self.confidence_factor * self.volatility_factor * \
               self.streak_factor * self.equity_curve_factor * market_factor * prediction_factor
        
        # Clamp risk within limits
        risk = min(self.max_risk, max(self.min_risk, risk))
        
        # Calculate position size based on risk
        risk_amount = self.current_capital * risk
        position_size = risk_amount / price
        
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percentage': risk,
            'factors': {
                'confidence': self.confidence_factor,
                'volatility': self.volatility_factor,
                'streak': self.streak_factor,
                'equity_curve': self.equity_curve_factor,
                'market': market_factor,
                'prediction_type': prediction_factor
            }
        }
        
    def update_after_trade(self, entry_price, exit_price, position_size, side='long', 
                         stop_loss=None, take_profit=None, timestamp=None):
        """
        Update position sizer with trade results
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            side: Trade side ('long' or 'short')
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            timestamp: Trade timestamp (optional)
            
        Returns:
            Trade PnL
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Calculate PnL
        if side == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - exit_price) * position_size
            
        # Update capital
        prev_capital = self.current_capital
        self.current_capital += pnl
        
        # Calculate drawdown
        if self.current_capital < prev_capital:
            drawdown = (prev_capital - self.current_capital) / prev_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
        # Update streak
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
            
        # Add to recent trades
        self.recent_trades_pnl.append(pnl)
        if len(self.recent_trades_pnl) > 10:
            self.recent_trades_pnl.pop(0)
            
        # Record trade
        trade = {
            'timestamp': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'side': side,
            'pnl': pnl,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'win': pnl > 0
        }
        self.trades.append(trade)
        
        return pnl
        
    def _calculate_confidence_factor(self, confidence):
        """Calculate factor based on model confidence"""
        # Scale confidence (0.5-1.0) to factor (0.5-1.5)
        return 0.5 + confidence
        
    def _calculate_volatility_factor(self, volatility):
        """Calculate factor based on market volatility"""
        # Inverse relationship with volatility (higher volatility = lower position size)
        return 1.0 / max(0.5, min(2.0, volatility))
        
    def _calculate_streak_factor(self):
        """Calculate factor based on winning/losing streak"""
        if self.win_streak > 0:
            # Increase risk slightly after wins (max +20% after 4 wins)
            return min(1.2, 1.0 + 0.05 * self.win_streak)
        elif self.loss_streak > 0:
            # Decrease risk after losses (min -40% after 4 losses)
            return max(0.6, 1.0 - 0.1 * self.loss_streak)
        return 1.0
        
    def _calculate_equity_curve_factor(self):
        """Calculate factor based on equity curve direction"""
        if not self.trades:
            return 1.0
            
        # Calculate equity curve direction over past 10 trades
        if len(self.trades) >= 10:
            recent_equity = [self.initial_capital]
            for trade in self.trades[-10:]:
                recent_equity.append(recent_equity[-1] + trade['pnl'])
            
            # Calculate linear regression slope
            x = np.arange(len(recent_equity))
            y = np.array(recent_equity)
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope to factor
            return min(1.2, max(0.8, 1.0 + slope / 1000))
            
        return 1.0
        
    def _get_prediction_type_factor(self, prediction_type):
        """Get factor for prediction type"""
        if prediction_type == 'trend':
            return 1.1  # Slightly more aggressive for trend following
        elif prediction_type == 'reversal':
            return 0.9  # More conservative for reversals
        elif prediction_type == 'breakout':
            return 1.2  # More aggressive for breakouts
        return 1.0
        
    def calculate_kelly_criterion(self, win_rate=None, win_loss_ratio=None):
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Win rate (if None, use historical)
            win_loss_ratio: Win/loss ratio (if None, use historical)
            
        Returns:
            Kelly criterion value
        """
        # Calculate historical win rate if not provided
        if win_rate is None:
            if not self.trades:
                return 0.5  # Default to half Kelly
            win_rate = sum(1 for trade in self.trades if trade['win']) / len(self.trades)
            
        # Calculate historical win/loss ratio if not provided
        if win_loss_ratio is None:
            if not self.trades:
                return 0.5  # Default to half Kelly
                
            wins = [trade['pnl'] for trade in self.trades if trade['win']]
            losses = [abs(trade['pnl']) for trade in self.trades if not trade['win']]
            
            if not losses:
                return 1.0  # All wins, use full Kelly
                
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 1
            
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            
        # Calculate Kelly percentage
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use half Kelly for more conservative sizing
        half_kelly = max(0, kelly_pct * 0.5)
        
        return half_kelly
        
    def calculate_position_with_kelly(self, price, confidence=1.0, volatility=1.0, 
                                     market_condition=None, prediction_type=None):
        """
        Calculate position size using Kelly criterion with adaptive adjustments
        
        Args:
            price: Current asset price
            confidence: Model confidence (0-1)
            volatility: Market volatility measure (1.0 = normal)
            market_condition: Current market condition (e.g., 'bull', 'bear')
            prediction_type: Type of prediction ('trend', 'reversal', etc.)
            
        Returns:
            Dictionary with position size information
        """
        # Calculate Kelly criterion
        kelly_pct = self.calculate_kelly_criterion()
        
        # Apply confidence factor to Kelly
        adjusted_kelly = kelly_pct * self.confidence_factor
        
        # Apply volatility factor
        adjusted_kelly *= self.volatility_factor
        
        # Apply market condition factor
        if market_condition:
            adjusted_kelly *= self.market_condition_factors.get(market_condition, 1.0)
            
        # Calculate position size based on Kelly
        position_size = (self.current_capital * adjusted_kelly) / price
        
        return {
            'position_size': position_size,
            'kelly_percentage': kelly_pct,
            'adjusted_kelly': adjusted_kelly,
            'risk_amount': self.current_capital * adjusted_kelly,
            'factors': {
                'confidence': self.confidence_factor,
                'volatility': self.volatility_factor,
                'market': self.market_condition_factors.get(market_condition, 1.0)
            }
        }
        
    def calculate_stop_loss(self, entry_price, side='long', volatility=1.0, atr=None):
        """
        Calculate adaptive stop loss level
        
        Args:
            entry_price: Entry price
            side: Trade side ('long' or 'short')
            volatility: Market volatility measure (1.0 = normal)
            atr: Average True Range (optional)
            
        Returns:
            Stop loss price
        """
        # Base stop distance (percentage)
        base_stop_pct = self.base_risk
        
        # Adjust for volatility
        if atr:
            # Use ATR-based stops
            stop_distance = atr * 2  # Default 2 ATR
            
            # Adjust for volatility
            stop_distance *= volatility
            
            # Set different stops based on side
            if side == 'long':
                return entry_price - stop_distance
            else:  # short
                return entry_price + stop_distance
        else:
            # Use percentage-based stops
            adjusted_stop_pct = base_stop_pct * volatility
            
            # Set different stops based on side
            if side == 'long':
                return entry_price * (1 - adjusted_stop_pct)
            else:  # short
                return entry_price * (1 + adjusted_stop_pct)
                
    def calculate_take_profit(self, entry_price, side='long', stop_loss=None, risk_reward=2.0):
        """
        Calculate adaptive take profit level
        
        Args:
            entry_price: Entry price
            side: Trade side ('long' or 'short')
            stop_loss: Stop loss price (optional)
            risk_reward: Risk-reward ratio (optional)
            
        Returns:
            Take profit price
        """
        if stop_loss:
            # Use risk-reward based take profit
            stop_distance = abs(entry_price - stop_loss)
            
            if side == 'long':
                return entry_price + (stop_distance * risk_reward)
            else:  # short
                return entry_price - (stop_distance * risk_reward)
        else:
            # Use simple percentage-based take profit
            take_profit_pct = self.base_risk * risk_reward
            
            if side == 'long':
                return entry_price * (1 + take_profit_pct)
            else:  # short
                return entry_price * (1 - take_profit_pct)
                
    def get_performance_stats(self):
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance stats
        """
        if not self.trades:
            return {
                'trade_count': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'max_drawdown': 0
            }
            
        wins = [trade['pnl'] for trade in self.trades if trade['win']]
        losses = [trade['pnl'] for trade in self.trades if not trade['win']]
        
        win_count = len(wins)
        loss_count = len(losses)
        total_count = len(self.trades)
        
        win_rate = win_count / total_count if total_count > 0 else 0
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        
        profit_factor = abs(total_wins / total_losses) if total_losses else float('inf')
        
        avg_win = total_wins / win_count if win_count > 0 else 0
        avg_loss = total_losses / loss_count if loss_count > 0 else 0
        
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        return {
            'trade_count': total_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'max_drawdown': self.max_drawdown,
            'current_capital': self.current_capital
        }
        
    def save(self, filename=None):
        """
        Save position sizing data to disk
        
        Args:
            filename: Filename to use (if None, use timestamp)
            
        Returns:
            Success status
        """
        try:
            if filename is None:
                filename = f"position_sizing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            filepath = os.path.join(self.save_path, filename)
            
            # Prepare data for serialization
            data = {
                'base_risk': self.base_risk,
                'max_risk': self.max_risk,
                'min_risk': self.min_risk,
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'recent_trades_pnl': self.recent_trades_pnl,
                'max_drawdown': self.max_drawdown,
                'market_condition_factors': self.market_condition_factors,
                'trades': [{
                    **trade,
                    'timestamp': trade['timestamp'].isoformat() if isinstance(trade['timestamp'], datetime) else trade['timestamp']
                } for trade in self.trades]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
                
            logging.info(f"Position sizing data saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving position sizing data: {e}")
            return False
            
    def load(self, filepath):
        """
        Load position sizing data from disk
        
        Args:
            filepath: Path to load data from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Update attributes
            self.base_risk = data['base_risk']
            self.max_risk = data['max_risk']
            self.min_risk = data['min_risk']
            self.initial_capital = data['initial_capital']
            self.current_capital = data['current_capital']
            self.win_streak = data['win_streak']
            self.loss_streak = data['loss_streak']
            self.recent_trades_pnl = data['recent_trades_pnl']
            self.max_drawdown = data['max_drawdown']
            self.market_condition_factors = data['market_condition_factors']
            
            # Parse trade timestamps
            self.trades = [{
                **trade,
                'timestamp': datetime.fromisoformat(trade['timestamp']) if isinstance(trade['timestamp'], str) else trade['timestamp']
            } for trade in data['trades']]
            
            logging.info(f"Position sizing data loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading position sizing data: {e}")
            return False

class AdaptiveRiskManager:
    """
    Advanced risk management system with dynamic adjustment
    """
    
    def __init__(self, position_sizer=None, max_open_positions=5, 
                max_risk_per_sector=0.2, correlation_threshold=0.7,
                save_path='models/risk_management'):
        """
        Initialize the adaptive risk manager
        
        Args:
            position_sizer: AdaptivePositionSizer instance
            max_open_positions: Maximum number of open positions
            max_risk_per_sector: Maximum risk per sector/asset class
            correlation_threshold: Correlation threshold for position limits
            save_path: Path to save risk management data
        """
        self.position_sizer = position_sizer or AdaptivePositionSizer()
        self.max_open_positions = max_open_positions
        self.max_risk_per_sector = max_risk_per_sector
        self.correlation_threshold = correlation_threshold
        self.save_path = save_path
        
        # Track open positions
        self.open_positions = []
        
        # Track risk exposure
        self.risk_exposure = {}
        
        # Correlation matrix
        self.correlation_matrix = {}
        
        # Risk limits
        self.risk_limits = {
            'daily_loss_limit': 0.03,  # 3% max daily loss
            'weekly_loss_limit': 0.07,  # 7% max weekly loss
            'max_drawdown_limit': 0.15  # 15% max drawdown
        }
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def check_risk_limits(self, new_position=None):
        """
        Check if a new position would violate risk limits
        
        Args:
            new_position: New position to check (optional)
            
        Returns:
            Dictionary with risk limit checks
        """
        # Get current risk exposure
        current_exposure = sum(pos['risk_amount'] for pos in self.open_positions)
        
        # Add new position risk if provided
        if new_position:
            current_exposure += new_position.get('risk_amount', 0)
            
        # Calculate capital risk
        capital = self.position_sizer.current_capital
        capital_risk = current_exposure / capital
        
        # Check maximum positions limit
        positions_limit_ok = len(self.open_positions) < self.max_open_positions
        
        # Check capital risk limit (max 20% at risk at any time)
        capital_risk_ok = capital_risk <= 0.2
        
        # Check sector risk limits
        sector_risk_ok = True
        if new_position and 'sector' in new_position:
            sector = new_position['sector']
            sector_exposure = sum(pos['risk_amount'] for pos in self.open_positions 
                                if pos.get('sector') == sector)
            sector_exposure += new_position.get('risk_amount', 0)
            sector_risk = sector_exposure / capital
            sector_risk_ok = sector_risk <= self.max_risk_per_sector
            
        # Check correlation risk
        correlation_ok = True
        if new_position and 'symbol' in new_position:
            symbol = new_position['symbol']
            for pos in self.open_positions:
                if 'symbol' in pos:
                    pair = (symbol, pos['symbol'])
                    if pair in self.correlation_matrix:
                        correlation = self.correlation_matrix[pair]
                        if correlation > self.correlation_threshold:
                            correlation_ok = False
                            break
                            
        # Check drawdown limit
        drawdown_ok = self.position_sizer.max_drawdown < self.risk_limits['max_drawdown_limit']
        
        # Overall assessment
        risk_ok = positions_limit_ok and capital_risk_ok and sector_risk_ok and correlation_ok and drawdown_ok
        
        return {
            'risk_ok': risk_ok,
            'reason': None if risk_ok else self._get_rejection_reason(positions_limit_ok, 
                      capital_risk_ok, sector_risk_ok, correlation_ok, drawdown_ok),
            'checks': {
                'positions_limit': positions_limit_ok,
                'capital_risk': capital_risk_ok,
                'sector_risk': sector_risk_ok,
                'correlation': correlation_ok,
                'drawdown': drawdown_ok
            },
            'exposure': {
                'current': current_exposure,
                'percentage': capital_risk
            }
        }
        
    def _get_rejection_reason(self, positions_limit_ok, capital_risk_ok, 
                            sector_risk_ok, correlation_ok, drawdown_ok):
        """Get rejection reason based on failed checks"""
        if not positions_limit_ok:
            return f"Maximum number of positions ({self.max_open_positions}) reached"
        elif not capital_risk_ok:
            return "Maximum capital at risk limit exceeded"
        elif not sector_risk_ok:
            return f"Maximum sector risk ({self.max_risk_per_sector * 100}%) exceeded"
        elif not correlation_ok:
            return f"High correlation with existing position detected"
        elif not drawdown_ok:
            return f"Maximum drawdown limit ({self.risk_limits['max_drawdown_limit'] * 100}%) reached"
        return "Risk limit exceeded"
        
    def add_position(self, position):
        """
        Add a position to the risk manager
        
        Args:
            position: Position dictionary
            
        Returns:
            Success status
        """
        # Check risk limits
        risk_check = self.check_risk_limits(position)
        
        if not risk_check['risk_ok']:
            logging.warning(f"Position rejected: {risk_check['reason']}")
            return False
            
        # Add position
        self.open_positions.append(position)
        
        # Update risk exposure
        sector = position.get('sector', 'unknown')
        if sector not in self.risk_exposure:
            self.risk_exposure[sector] = 0
        self.risk_exposure[sector] += position.get('risk_amount', 0)
        
        return True
        
    def close_position(self, position_id):
        """
        Close a position
        
        Args:
            position_id: ID of position to close
            
        Returns:
            Closed position or None
        """
        for i, position in enumerate(self.open_positions):
            if position.get('id') == position_id:
                closed_position = self.open_positions.pop(i)
                
                # Update risk exposure
                sector = closed_position.get('sector', 'unknown')
                if sector in self.risk_exposure:
                    self.risk_exposure[sector] -= closed_position.get('risk_amount', 0)
                    
                return closed_position
                
        return None
        
    def update_correlation_matrix(self, correlation_data):
        """
        Update correlation matrix
        
        Args:
            correlation_data: Dictionary with correlation data
        """
        self.correlation_matrix = correlation_data
        
    def update_risk_limits(self, limits):
        """
        Update risk limits
        
        Args:
            limits: Dictionary with risk limits
        """
        self.risk_limits.update(limits)
        
    def get_portfolio_var(self, confidence=0.95):
        """
        Calculate portfolio Value at Risk (VaR)
        
        Args:
            confidence: Confidence level
            
        Returns:
            VaR value
        """
        if not self.open_positions:
            return 0
            
        # Get position values
        position_values = [pos.get('current_value', 0) for pos in self.open_positions]
        total_value = sum(position_values)
        
        # Get position weights
        weights = [value / total_value for value in position_values]
        
        # Get position volatilities
        volatilities = [pos.get('volatility', 0.02) for pos in self.open_positions]
        
        # Calculate portfolio volatility (simplified)
        portfolio_variance = 0
        for i, w_i in enumerate(weights):
            for j, w_j in enumerate(weights):
                # Get correlation between assets i and j
                corr = 1.0 if i == j else 0.5  # Default correlation
                
                # Get actual correlation if available
                if i != j:
                    symbol_i = self.open_positions[i].get('symbol')
                    symbol_j = self.open_positions[j].get('symbol')
                    if symbol_i and symbol_j:
                        pair = (symbol_i, symbol_j)
                        if pair in self.correlation_matrix:
                            corr = self.correlation_matrix[pair]
                
                portfolio_variance += w_i * w_j * volatilities[i] * volatilities[j] * corr
                
        portfolio_volatility = math.sqrt(portfolio_variance)
        
        # Calculate VaR
        z_score = 1.645  # For 95% confidence
        if confidence == 0.99:
            z_score = 2.326  # For 99% confidence
            
        daily_var = total_value * portfolio_volatility * z_score
        
        return daily_var
        
    def recommend_portfolio_adjustment(self):
        """
        Recommend portfolio adjustments to optimize risk
        
        Returns:
            List of recommended adjustments
        """
        if not self.open_positions:
            return []
            
        recommendations = []
        
        # Check overall portfolio risk
        var = self.get_portfolio_var()
        capital = self.position_sizer.current_capital
        var_pct = var / capital
        
        if var_pct > 0.03:  # If daily VaR > 3%
            recommendations.append({
                'type': 'reduce_risk',
                'message': f"Portfolio VaR ({var_pct:.2%}) exceeds target. Consider reducing position sizes.",
                'severity': 'high' if var_pct > 0.05 else 'medium'
            })
            
        # Check sector concentration
        for sector, exposure in self.risk_exposure.items():
            sector_pct = exposure / capital
            if sector_pct > self.max_risk_per_sector:
                recommendations.append({
                    'type': 'sector_concentration',
                    'message': f"High concentration in {sector} sector ({sector_pct:.2%}). Consider diversifying.",
                    'severity': 'medium',
                    'sector': sector
                })
                
        # Check correlation
        high_corr_pairs = []
        for i, pos_i in enumerate(self.open_positions):
            for j, pos_j in enumerate(self.open_positions):
                if i < j:  # Avoid duplicates
                    symbol_i = pos_i.get('symbol')
                    symbol_j = pos_j.get('symbol')
                    if symbol_i and symbol_j:
                        pair = (symbol_i, symbol_j)
                        if pair in self.correlation_matrix:
                            corr = self.correlation_matrix[pair]
                            if corr > self.correlation_threshold:
                                high_corr_pairs.append((symbol_i, symbol_j, corr))
                                
        if high_corr_pairs:
            for symbol_i, symbol_j, corr in high_corr_pairs:
                recommendations.append({
                    'type': 'high_correlation',
                    'message': f"High correlation ({corr:.2f}) between {symbol_i} and {symbol_j}. Consider reducing exposure.",
                    'severity': 'medium',
                    'symbols': [symbol_i, symbol_j]
                })
                
        return recommendations
        
    def save(self, filename=None):
        """
        Save risk management data to disk
        
        Args:
            filename: Filename to use (if None, use timestamp)
            
        Returns:
            Success status
        """
        try:
            if filename is None:
                filename = f"risk_management_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            filepath = os.path.join(self.save_path, filename)
            
            # Prepare data for serialization
            data = {
                'max_open_positions': self.max_open_positions,
                'max_risk_per_sector': self.max_risk_per_sector,
                'correlation_threshold': self.correlation_threshold,
                'risk_limits': self.risk_limits,
                'risk_exposure': self.risk_exposure,
                'open_positions': [{
                    **pos,
                    'entry_time': pos.get('entry_time').isoformat() if pos.get('entry_time') and isinstance(pos.get('entry_time'), datetime) else pos.get('entry_time')
                } for pos in self.open_positions],
                'correlation_matrix': self.correlation_matrix
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
                
            logging.info(f"Risk management data saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving risk management data: {e}")
            return False
            
    def load(self, filepath):
        """
        Load risk management data from disk
        
        Args:
            filepath: Path to load data from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Update attributes
            self.max_open_positions = data['max_open_positions']
            self.max_risk_per_sector = data['max_risk_per_sector']
            self.correlation_threshold = data['correlation_threshold']
            self.risk_limits = data['risk_limits']
            self.risk_exposure = data['risk_exposure']
            self.correlation_matrix = data['correlation_matrix']
            
            # Parse position timestamps
            self.open_positions = [{
                **pos,
                'entry_time': datetime.fromisoformat(pos['entry_time']) if pos.get('entry_time') and isinstance(pos['entry_time'], str) else pos.get('entry_time')
            } for pos in data['open_positions']]
            
            logging.info(f"Risk management data loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading risk management data: {e}")
            return False
