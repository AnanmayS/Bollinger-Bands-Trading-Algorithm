"""
Script to compare different Bollinger Bands strategy configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import fetch_data, calculate_performance_metrics, print_performance_metrics
from bollinger_strategy import BollingerBandsStrategy

def compare_strategies(ticker, start_date, end_date, initial_capital=10000.0, strategies=None):
    """
    Compare multiple Bollinger Bands strategy configurations.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol of the stock
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    initial_capital : float, optional
        Initial investment amount (default: 10000.0)
    strategies : list, optional
        List of dictionaries with strategy configurations
        Each dictionary should have 'name', 'window', and 'num_std' keys
    """
    # Default strategies if none provided
    if strategies is None:
        strategies = [
            {'name': 'Default (20, 2.0)', 'window': 20, 'num_std': 2.0},
            {'name': 'Narrow Bands (20, 1.5)', 'window': 20, 'num_std': 1.5},
            {'name': 'Wide Bands (20, 2.5)', 'window': 20, 'num_std': 2.5},
            {'name': 'Short Window (10, 2.0)', 'window': 10, 'num_std': 2.0},
            {'name': 'Long Window (50, 2.0)', 'window': 50, 'num_std': 2.0}
        ]
    
    # Fetch historical data
    print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    data = fetch_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        print("No data available. Exiting.")
        return
    
    # Ensure data is clean and properly formatted
    data = data.copy()
    data.dropna(inplace=True)
    
    print(f"Data fetched successfully. {len(data)} data points.")
    
    # Run backtest for each strategy
    results = []
    portfolio_values = pd.DataFrame(index=data.index)
    
    for strategy_config in strategies:
        name = strategy_config['name']
        window = strategy_config['window']
        num_std = strategy_config['num_std']
        
        print(f"\nRunning backtest for strategy: {name}")
        strategy = BollingerBandsStrategy(window=window, num_std=num_std)
        
        try:
            positions, signal_points = strategy.backtest(data, initial_capital)
            
            # Calculate performance metrics
            returns = positions['strategy_returns'].dropna()
            final_capital = positions['portfolio_value'].iloc[-1]
            metrics = calculate_performance_metrics(initial_capital, final_capital, returns)
            
            # Add to results
            metrics['name'] = name
            metrics['window'] = window
            metrics['num_std'] = num_std
            metrics['final_capital'] = final_capital
            results.append(metrics)
            
            # Add portfolio value to comparison DataFrame
            portfolio_values[name] = positions['portfolio_value']
        except Exception as e:
            print(f"Error running backtest for strategy {name}: {e}")
            continue
    
    if not results:
        print("No successful backtests. Exiting.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS".center(80))
    print("="*80)
    print(results_df[['name', 'Total Return', 'Annualized Return', 'Sharpe Ratio', 'Maximum Drawdown', 'Win Rate']].to_string(index=False))
    print("="*80)
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    for column in portfolio_values.columns:
        plt.plot(portfolio_values.index, portfolio_values[column], label=column)
    
    plt.title(f'Strategy Comparison - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{ticker}_strategy_comparison.png')
    plt.show()
    
    # Save results to CSV
    results_df.to_csv(f'{ticker}_strategy_comparison.csv', index=False)
    print(f"Results saved to {ticker}_strategy_comparison.csv and {ticker}_strategy_comparison.png")
    
    return results_df, portfolio_values

if __name__ == "__main__":
    # Example: Compare different Bollinger Bands strategies on SPY
    compare_strategies(
        ticker='SPY',
        start_date='2018-01-01',
        end_date='2023-12-31',
        initial_capital=10000.0
    ) 