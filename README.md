# Bollinger Bands Trading Algorithm

A sophisticated algorithmic trading system that implements the Bollinger Bands strategy with an interactive web interface for backtesting and visualization.

![Bollinger Bands Strategy](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Bollinger_Bands.svg/1200px-Bollinger_Bands.svg.png)

## Overview

This project provides a complete framework for backtesting a Bollinger Bands trading strategy on historical stock data. It includes:

- A robust implementation of the Bollinger Bands trading strategy
- Interactive web interface built with Streamlit
- Comprehensive backtesting capabilities
- Performance metrics and visualization
- Parameter optimization
- Strategy comparison tools

## Features

- **Bollinger Bands Strategy**: Implements a mean-reversion strategy based on Bollinger Bands technical indicator
- **Interactive Web Interface**: User-friendly Streamlit app for running backtests and visualizing results
- **Historical Data**: Fetches and processes historical stock data using yfinance
- **Backtesting Engine**: Simulates trading with customizable parameters
- **Performance Metrics**: Calculates key metrics including:
  - Total Return
  - Annualized Return
  - Sharpe Ratio
  - Win Rate
  - Maximum Drawdown
  - Annualized Volatility
- **Parameter Optimization**: Finds optimal strategy parameters using grid search
- **Strategy Comparison**: Compare different parameter configurations side-by-side
- **Visualization**: Interactive charts for portfolio value, price, and trading signals

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bollinger-bands-trading.git
   cd bollinger-bands-trading
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface

Launch the Streamlit app:
```
streamlit run app.py
```

The web interface allows you to:
- Select a stock ticker
- Choose a date range for backtesting
- Set strategy parameters (window size and standard deviation)
- Run parameter optimization
- Compare different strategy configurations
- View detailed performance metrics and visualizations

### Running from Command Line

For command-line backtesting:
```
python bollinger_backtest.py --ticker AAPL --start 2020-01-01 --end 2023-01-01 --window 20 --std 2.0
```

For strategy comparison:
```
python compare_strategies.py --ticker MSFT --start 2020-01-01 --end 2023-01-01
```

## Strategy Overview

The Bollinger Bands strategy is a technical analysis tool that consists of:
- A middle band (typically a 20-day simple moving average)
- An upper band (middle band + 2 standard deviations)
- A lower band (middle band - 2 standard deviations)

The strategy generates:
- Buy signals when the price crosses below the lower band and then crosses back above it
- Sell signals when the price crosses above the upper band and then crosses back below it

This implementation focuses on long-only positions (no shorting).

## Project Structure

- `app.py`: Streamlit web application
- `bollinger_strategy.py`: Core implementation of the Bollinger Bands strategy
- `bollinger_backtest.py`: Command-line script for running backtests
- `compare_strategies.py`: Script for comparing different strategy configurations
- `utils.py`: Utility functions for data processing and visualization
- `requirements.txt`: Project dependencies
- `example.py`: Example usage of the strategy

## Performance Considerations

The strategy's performance varies significantly across different stocks and market conditions:
- Works best in range-bound markets
- May underperform in strong trending markets
- Parameter optimization is crucial for adapting to different assets

## Future Improvements

- Add more technical indicators (RSI, MACD, etc.)
- Implement portfolio optimization for multiple assets
- Add risk management features (stop-loss, position sizing)
- Integrate with live trading APIs
- Implement machine learning for parameter selection
- Add more sophisticated performance metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for providing historical market data
- [Streamlit](https://streamlit.io/) for the interactive web interface
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data processing
- [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/) for visualization 