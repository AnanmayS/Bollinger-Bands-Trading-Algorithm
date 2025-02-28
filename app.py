import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import datetime
import yfinance as yf

from bollinger_strategy import BollingerBandsStrategy
from utils import fetch_data, calculate_performance_metrics
from bollinger_backtest import run_backtest

# Set page configuration
st.set_page_config(
    page_title="Bollinger Bands Trading Algorithm",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Bollinger Bands Trading Algorithm")
st.markdown("""
This app allows you to backtest a Bollinger Bands trading strategy on various stocks.
You can customize the parameters or use optimization to find the best parameters.
""")

# Sidebar for inputs
st.sidebar.header("Strategy Parameters")

# Stock selection
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")

# Date range selection
today = datetime.date.today()
start_date = st.sidebar.date_input(
    "Start Date",
    today - datetime.timedelta(days=365*3)  # 3 years ago
)
end_date = st.sidebar.date_input("End Date", today)

# Convert dates to string format
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Initial capital
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

# Strategy parameters
use_optimization = st.sidebar.checkbox("Use Parameter Optimization", value=False)

if not use_optimization:
    window = st.sidebar.slider(
        "Window Size (days)",
        min_value=5,
        max_value=50,
        value=20,
        step=1
    )
    
    num_std = st.sidebar.slider(
        "Number of Standard Deviations",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )
else:
    st.sidebar.info("Parameters will be optimized automatically.")
    window = 20  # Default value, will be optimized
    num_std = 2.0  # Default value, will be optimized

# Run backtest button
run_button = st.sidebar.button("Run Backtest")

# Function to create static image for backtest results
def create_backtest_image(positions, signal_points, bb_data, ticker):
    """
    Create a static matplotlib image for backtest results
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
    fig.set_facecolor('white')
    
    # Plot portfolio value in the first subplot
    ax1.plot(positions.index, positions['portfolio_value'], 'b-', linewidth=2, label='Portfolio Value')
    ax1.set_title(f'Portfolio Value Over Time - {ticker}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot price and Bollinger Bands in the second subplot
    ax2.plot(bb_data.index, bb_data['Close'], 'k-', linewidth=1.5, label='Close Price')
    ax2.plot(bb_data.index, bb_data['middle_band'], 'b-', linewidth=1.5, label='Middle Band (SMA)')
    ax2.plot(bb_data.index, bb_data['upper_band'], 'r-', linewidth=1.5, label='Upper Band')
    ax2.plot(bb_data.index, bb_data['lower_band'], 'g-', linewidth=1.5, label='Lower Band')
    
    # Plot buy signals
    buy_signals = signal_points[signal_points['signal'] == 1]
    if not buy_signals.empty:
        ax2.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color='green', s=150, label='Buy Signal', zorder=5,
                   edgecolors='darkgreen', linewidth=1.5)
    
    # Plot sell signals
    sell_signals = signal_points[signal_points['signal'] == -1]
    if not sell_signals.empty:
        ax2.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color='red', s=150, label='Sell Signal', zorder=5,
                   edgecolors='darkred', linewidth=1.5)
    
    ax2.set_title(f'Bollinger Bands Strategy - {ticker}', fontsize=14)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Function to display performance metrics
def display_metrics(metrics):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{metrics['Total Return']:.2%}")
        st.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
    
    with col3:
        st.metric("Maximum Drawdown", f"{metrics['Maximum Drawdown']:.2%}")
        st.metric("Annualized Volatility", f"{metrics['Annualized Volatility']:.2%}")

# Function to highlight position status in dataframe
def highlight_position_status(val):
    """
    Highlight position status with colors
    """
    if val == 'BUY & HOLD':
        return 'background-color: lightgreen; color: darkgreen; font-weight: bold'
    elif val == 'HOLDING':
        return 'background-color: #e6ffe6; color: green'
    elif val == 'SELL':
        return 'background-color: #ffcccc; color: darkred; font-weight: bold'
    elif val == 'OUT OF MARKET':
        return 'background-color: #f2f2f2; color: gray'
    return ''

# Main app logic
if run_button:
    # Show loading spinner
    with st.spinner(f"Fetching data and running backtest for {ticker}..."):
        try:
            # Fetch data
            data = fetch_data(ticker, start_date_str, end_date_str)
            
            if data is None or data.empty:
                st.error(f"No data available for {ticker} between {start_date_str} and {end_date_str}")
            else:
                # Ensure data is clean
                data = data.copy()
                data.dropna(inplace=True)
                
                # Initialize strategy
                strategy = BollingerBandsStrategy(window=window, num_std=num_std)
                
                # Optimize if requested
                if use_optimization:
                    st.info("Optimizing strategy parameters...")
                    best_window, best_num_std, best_sharpe = strategy.optimize(data)
                    st.success(f"Optimization complete. Best parameters: window={best_window}, num_std={best_num_std:.1f}, Sharpe={best_sharpe:.2f}")
                
                # Run backtest
                positions, signal_points = strategy.backtest(data, initial_capital)
                
                # Calculate performance metrics
                returns = positions['strategy_returns'].dropna()
                final_capital = positions['portfolio_value'].iloc[-1]
                metrics = calculate_performance_metrics(initial_capital, final_capital, returns)
                
                # Display metrics
                st.header("Performance Metrics")
                display_metrics(metrics)
                
                # Prepare data for plotting
                bb_data = data.copy()
                bb_data['middle_band'] = bb_data['Close'].rolling(window=strategy.window).mean()
                bb_data['std'] = bb_data['Close'].rolling(window=strategy.window).std()
                bb_data['upper_band'] = bb_data['middle_band'] + (bb_data['std'] * strategy.num_std)
                bb_data['lower_band'] = bb_data['middle_band'] - (bb_data['std'] * strategy.num_std)
                bb_data = bb_data.dropna().copy()
                
                # Create and display static image
                st.header("Backtest Results")
                img_buf = create_backtest_image(positions, signal_points, bb_data, ticker)
                st.image(img_buf, use_container_width=True)
                
                # Display positions table with highlighted position status
                st.header("Trading Positions")
                
                # Select columns to display
                display_cols = ['price', 'signal', 'position', 'position_status', 
                               'strategy_returns', 'portfolio_value', 'holdings', 'cash']
                
                # Apply styling to highlight position status
                styled_positions = positions[display_cols].style.map(
                    highlight_position_status, subset=['position_status']
                )
                
                st.dataframe(styled_positions)
                
                # Display trade summary
                st.header("Trade Summary")
                
                # Count trades
                buy_signals = positions[positions['position_status'] == 'BUY & HOLD']
                sell_signals = positions[positions['position_status'] == 'SELL']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", len(buy_signals))
                with col2:
                    st.metric("Days In Market", positions['position'].sum())
                with col3:
                    st.metric("Market Exposure", f"{positions['position'].mean():.2%}")
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = positions.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"{ticker}_backtest_results.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Add strategy comparison section
st.header("Strategy Comparison")
st.markdown("""
Compare different Bollinger Bands configurations on the same stock.
Select the configurations you want to compare.
""")

# Strategy comparison inputs
compare_ticker = st.text_input("Stock Ticker for Comparison", ticker)
compare_start_date = st.date_input(
    "Comparison Start Date",
    start_date
)
compare_end_date = st.date_input(
    "Comparison End Date",
    end_date
)

# Convert dates to string format
compare_start_date_str = compare_start_date.strftime('%Y-%m-%d')
compare_end_date_str = compare_end_date.strftime('%Y-%m-%d')

# Strategy configurations to compare
st.subheader("Select Configurations to Compare")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    compare_default = st.checkbox("Default (20, 2.0)", value=True)
with col2:
    compare_narrow = st.checkbox("Narrow Bands (20, 1.5)", value=True)
with col3:
    compare_wide = st.checkbox("Wide Bands (20, 2.5)", value=True)
with col4:
    compare_short = st.checkbox("Short Window (10, 2.0)", value=True)
with col5:
    compare_long = st.checkbox("Long Window (50, 2.0)", value=True)

# Run comparison button
compare_button = st.button("Run Comparison")

# Function to create comparison image
def create_comparison_image(portfolio_values, ticker):
    """
    Create a static matplotlib image for strategy comparison
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor('white')
    
    # Define a list of colors for different strategies
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, column in enumerate(portfolio_values.columns):
        color_idx = i % len(colors)
        ax.plot(portfolio_values.index, portfolio_values[column], 
                color=colors[color_idx], linewidth=2, label=column)
    
    ax.set_title(f'Strategy Comparison - {ticker}', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Run comparison logic
if compare_button:
    with st.spinner(f"Running comparison for {compare_ticker}..."):
        try:
            # Fetch data
            data = fetch_data(compare_ticker, compare_start_date_str, compare_end_date_str)
            
            if data is None or data.empty:
                st.error(f"No data available for {compare_ticker} between {compare_start_date_str} and {compare_end_date_str}")
            else:
                # Ensure data is clean
                data = data.copy()
                data.dropna(inplace=True)
                
                # Create list of strategies to compare
                strategies = []
                if compare_default:
                    strategies.append({'name': 'Default (20, 2.0)', 'window': 20, 'num_std': 2.0})
                if compare_narrow:
                    strategies.append({'name': 'Narrow Bands (20, 1.5)', 'window': 20, 'num_std': 1.5})
                if compare_wide:
                    strategies.append({'name': 'Wide Bands (20, 2.5)', 'window': 20, 'num_std': 2.5})
                if compare_short:
                    strategies.append({'name': 'Short Window (10, 2.0)', 'window': 10, 'num_std': 2.0})
                if compare_long:
                    strategies.append({'name': 'Long Window (50, 2.0)', 'window': 50, 'num_std': 2.0})
                
                if not strategies:
                    st.warning("Please select at least one strategy configuration to compare.")
                else:
                    # Run backtest for each strategy
                    results = []
                    portfolio_values = pd.DataFrame(index=data.index)
                    
                    for strategy_config in strategies:
                        name = strategy_config['name']
                        window = strategy_config['window']
                        num_std = strategy_config['num_std']
                        
                        st.text(f"Running backtest for strategy: {name}")
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
                            metrics['trades'] = len(positions[positions['position_status'] == 'BUY & HOLD'])
                            metrics['market_exposure'] = positions['position'].mean()
                            results.append(metrics)
                            
                            # Add portfolio value to comparison DataFrame
                            portfolio_values[name] = positions['portfolio_value']
                        except Exception as e:
                            st.error(f"Error running backtest for strategy {name}: {e}")
                            continue
                    
                    if not results:
                        st.error("No successful backtests. Please try different parameters.")
                    else:
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Comparison Results")
                        st.dataframe(results_df[['name', 'Total Return', 'Annualized Return', 
                                               'Sharpe Ratio', 'Maximum Drawdown', 'Win Rate',
                                               'trades', 'market_exposure']])
                        
                        # Create and display static image
                        img_buf = create_comparison_image(portfolio_values, compare_ticker)
                        st.image(img_buf, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Comparison Results as CSV",
                            data=csv,
                            file_name=f"{compare_ticker}_strategy_comparison.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"An error occurred during comparison: {e}")

# Add information about the strategy
with st.expander("About Bollinger Bands Strategy"):
    st.markdown("""
    ## Bollinger Bands Strategy

    Bollinger Bands are a technical analysis tool developed by John Bollinger in the 1980s. They consist of:
    
    - A middle band (typically a 20-day simple moving average)
    - An upper band (middle band + N standard deviations)
    - A lower band (middle band - N standard deviations)
    
    ### How the Strategy Works
    
    This implementation generates:
    - **Buy signals** when the price crosses below the lower band and then crosses back above it
    - **Sell signals** when the price crosses above the upper band and then crosses back below it
    
    The strategy is based on the concept of mean reversion, where prices tend to return to their average over time.
    
    ### Position Status Explained
    
    - **BUY & HOLD**: A buy signal was generated and a position was entered
    - **HOLDING**: Currently holding a position (already in the market)
    - **SELL**: A sell signal was generated and a position was exited
    - **OUT OF MARKET**: No position is held (in cash)
    
    ### Parameters
    
    - **Window Size**: The number of days used for the moving average (default: 20)
    - **Number of Standard Deviations**: Determines the width of the bands (default: 2)
    
    ### Performance Metrics
    
    - **Total Return**: The overall return of the strategy
    - **Annualized Return**: The return normalized to a yearly rate
    - **Sharpe Ratio**: Measures risk-adjusted return (higher is better)
    - **Maximum Drawdown**: The largest peak-to-trough decline
    - **Win Rate**: The percentage of profitable trades
    - **Market Exposure**: Percentage of time invested in the market
    """)
