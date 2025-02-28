"""
Example script demonstrating how to use the Bollinger Bands backtesting framework.
"""

from bollinger_backtest import run_backtest

if __name__ == "__main__":
    # Example 1: Basic backtest with default parameters
    print("Example 1: Basic backtest of TSLA with default parameters")
    run_backtest(
        ticker='TSLA',
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=10000.0
    )
    
    # Example 2: Backtest with custom parameters
    print("\n\nExample 2: Backtest of MSFT with custom parameters")
    run_backtest(
        ticker='MSFT',
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=10000.0,
        window=15,
        num_std=2.5
    )
    
    # Example 3: Backtest with parameter optimization
    print("\n\nExample 3: Backtest of AMZN with parameter optimization")
    run_backtest(
        ticker='AMZN',
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=10000.0,
        optimize=True
    ) 