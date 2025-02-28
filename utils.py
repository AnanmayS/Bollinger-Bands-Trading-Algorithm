import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

def fetch_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical market data for a given ticker.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol of the stock
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    interval : str, optional
        Data interval (default: '1d' for daily)
        
    Returns:
    --------
    pandas.DataFrame
        Historical market data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def plot_bollinger_bands(data, ticker, signals=None):
    """
    Plot price data with Bollinger Bands and optional buy/sell signals.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with price data and Bollinger Bands
    ticker : str
        Ticker symbol for the title
    signals : pandas.DataFrame, optional
        DataFrame containing buy/sell signals
    """
    plt.figure(figsize=(12, 6))
    
    # Plot close price
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
    
    # Plot Bollinger Bands
    plt.plot(data.index, data['middle_band'], label='Middle Band (SMA)', color='blue', alpha=0.7)
    plt.plot(data.index, data['upper_band'], label='Upper Band', color='red', alpha=0.7)
    plt.plot(data.index, data['lower_band'], label='Lower Band', color='green', alpha=0.7)
    
    # Plot buy/sell signals if provided
    if signals is not None:
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color='green', s=100, label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'Bollinger Bands for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def calculate_performance_metrics(initial_capital, final_capital, returns, risk_free_rate=0.02, trading_days=252):
    """
    Calculate performance metrics for a trading strategy.
    
    Parameters:
    -----------
    initial_capital : float
        Initial investment amount
    final_capital : float
        Final portfolio value
    returns : pandas.Series
        Daily returns of the strategy
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%)
    trading_days : int, optional
        Number of trading days in a year (default: 252)
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Total return
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Annualized return
    period_years = len(returns) / trading_days
    
    # Handle negative total returns for annualized calculation
    if total_return <= -1:
        annualized_return = -1.0  # Complete loss
    else:
        annualized_return = (1 + total_return) ** (1 / period_years) - 1
    
    # Volatility (annualized standard deviation)
    daily_std = returns.std()
    annualized_std = daily_std * np.sqrt(trading_days)
    
    # Sharpe ratio
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = returns - daily_risk_free
    
    # Avoid division by zero
    if daily_std > 0:
        sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(trading_days)
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_std,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate
    }
    
    return metrics

def print_performance_metrics(metrics):
    """
    Print performance metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing performance metrics
    """
    print("\n" + "="*50)
    print("PERFORMANCE METRICS".center(50))
    print("="*50)
    
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Annualized Return: {metrics['Annualized Return']:.2%}")
    print(f"Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['Maximum Drawdown']:.2%}")
    print(f"Win Rate: {metrics['Win Rate']:.2%}")
    
    print("="*50 + "\n") 