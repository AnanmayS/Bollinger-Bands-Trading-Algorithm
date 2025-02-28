import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

from utils import fetch_data, plot_bollinger_bands, calculate_performance_metrics, print_performance_metrics
from bollinger_strategy import BollingerBandsStrategy

def run_backtest(ticker, start_date, end_date, initial_capital=10000.0, window=20, num_std=2, optimize=False):
    """
    Run a backtest of the Bollinger Bands strategy.
    
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
    window : int, optional
        Window size for moving average (default: 20)
    num_std : int, optional
        Number of standard deviations for bands (default: 2)
    optimize : bool, optional
        Whether to optimize strategy parameters (default: False)
    """
    # Fetch historical data
    print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    data = fetch_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        print("No data available. Exiting.")
        return
    
    print(f"Data fetched successfully. {len(data)} data points.")
    
    # Ensure data is clean and properly formatted
    data = data.copy()
    data.dropna(inplace=True)
    
    # Initialize strategy
    strategy = BollingerBandsStrategy(window=window, num_std=num_std)
    
    # Optimize parameters if requested
    if optimize:
        print("Optimizing strategy parameters...")
        best_window, best_num_std, best_sharpe = strategy.optimize(data)
        print(f"Optimization complete. Best parameters: window={best_window}, num_std={best_num_std:.1f}, Sharpe={best_sharpe:.2f}")
    
    # Run backtest
    print("Running backtest...")
    positions, signal_points = strategy.backtest(data, initial_capital)
    
    # Calculate performance metrics
    returns = positions['strategy_returns'].dropna()
    final_capital = positions['portfolio_value'].iloc[-1]
    metrics = calculate_performance_metrics(initial_capital, final_capital, returns)
    
    # Print performance metrics
    print_performance_metrics(metrics)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(positions.index, positions['portfolio_value'])
    plt.title(f'Portfolio Value Over Time - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot Bollinger Bands with signals
    plt.subplot(2, 1, 2)
    bb_data = data.copy()
    bb_data['middle_band'] = bb_data['Close'].rolling(window=strategy.window).mean()
    bb_data['std'] = bb_data['Close'].rolling(window=strategy.window).std()
    bb_data['upper_band'] = bb_data['middle_band'] + (bb_data['std'] * strategy.num_std)
    bb_data['lower_band'] = bb_data['middle_band'] - (bb_data['std'] * strategy.num_std)
    
    # Ensure data is aligned
    bb_data = bb_data.dropna().copy()
    
    # Plot price and bands
    plt.plot(bb_data.index, bb_data['Close'], label='Close Price', alpha=0.5)
    plt.plot(bb_data.index, bb_data['middle_band'], label='Middle Band (SMA)', color='blue', alpha=0.7)
    plt.plot(bb_data.index, bb_data['upper_band'], label='Upper Band', color='red', alpha=0.7)
    plt.plot(bb_data.index, bb_data['lower_band'], label='Lower Band', color='green', alpha=0.7)
    
    # Plot buy/sell signals
    buy_signals = signal_points[signal_points['signal'] == 1]
    sell_signals = signal_points[signal_points['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], 
               marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['Close'], 
               marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'Bollinger Bands Strategy - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_backtest_results.png')
    plt.show()
    
    # Save results to CSV
    positions.to_csv(f'{ticker}_backtest_results.csv')
    print(f"Results saved to {ticker}_backtest_results.csv and {ticker}_backtest_results.png")
    
    return positions, signal_points, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest Bollinger Bands trading strategy')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--window', type=int, default=20, help='Window size for moving average')
    parser.add_argument('--num_std', type=float, default=2.0, help='Number of standard deviations for bands')
    parser.add_argument('--optimize', action='store_true', help='Optimize strategy parameters')
    
    args = parser.parse_args()
    
    run_backtest(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        window=args.window,
        num_std=args.num_std,
        optimize=args.optimize
    ) 