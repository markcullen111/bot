# Automated Trading System

A professional-grade automated trading system built with Python, supporting multiple brokers, strategies, and real-time monitoring.

## Features

- **Production-Ready**: Built for real-world trading with proper error handling and logging
- **Multi-Broker Support**: Currently supports Interactive Brokers with an extensible interface for other brokers
- **Risk Management**: Comprehensive risk management with position sizing and drawdown protection
- **Strategy Framework**: Flexible strategy implementation with mean reversion and trend following examples
- **Real-Time Monitoring**: Prometheus metrics and JSON logging for system monitoring
- **Professional UI**: PyQt5-based GUI for real-time trading visualization

## Prerequisites

- Python 3.8+
- Interactive Brokers TWS or IB Gateway
- PostgreSQL (optional, for database storage)
- Docker (optional, for containerization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-system.git
cd trading-system
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create configuration:
```bash
cp config.example.json config.json
```

5. Edit `config.json` with your settings:
- Add your broker API credentials
- Configure risk parameters
- Set up trading strategies
- Configure symbols to trade

## Configuration

The system is configured via `config.json`. Key configuration sections:

- `broker`: Broker connection settings and API credentials
- `risk_limits`: Risk management parameters
- `market_data`: Data sources and symbols
- `strategies`: Trading strategy configurations
- `execution`: Order execution settings
- `logging`: Logging and monitoring configuration

## Usage

1. Start the trading system:
```python
from trading_system import TradingSystem

# Initialize the system
trading_system = TradingSystem(config_path="config.json")

# Run a strategy
trading_system.run_strategy("mean_reversion")
```

2. Monitor the system:
- Check logs in the `logs` directory
- View Prometheus metrics at `http://localhost:8000`
- Watch the trading GUI for real-time updates

## Project Structure

```
trading-system/
├── core/
│   ├── brokers/         # Broker implementations
│   └── logging_config.py # Logging configuration
├── strategies/          # Trading strategies
├── risk_management/     # Risk management components
├── gui/                 # PyQt5 GUI components
├── data/               # Data handling
├── analysis/           # Market analysis tools
├── execution/          # Order execution
├── config.json         # Configuration file
└── trading_system.py   # Main system implementation
```

## Risk Management

The system includes comprehensive risk management:
- Position sizing based on account value
- Maximum drawdown protection
- Per-trade risk limits
- Daily loss limits
- Maximum leverage control

## Monitoring

- JSON-formatted logs with rotation
- Prometheus metrics for:
  - Order counts
  - Position values
  - Account balance
  - Error counts
- Real-time GUI updates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this system.
