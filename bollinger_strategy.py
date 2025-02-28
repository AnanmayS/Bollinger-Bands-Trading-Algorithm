import pandas as pd
import numpy as np

class BollingerBandsStrategy:
    """
    A trading strategy based on Bollinger Bands.
    
    This strategy generates:
    - Buy signals when the price crosses below the lower band and then crosses back above it
    - Sell signals (exit positions) when the price crosses above the upper band and then crosses back below it
    - No shorting is allowed in this implementation
    """
    
    def __init__(self, window=20, num_std=2):
        """
        Initialize the Bollinger Bands strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window size for moving average (default: 20)
        num_std : int, optional
            Number of standard deviations for bands (default: 2)
        """
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with 'Close' column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added signal column (1 for buy, -1 for sell/exit, 0 for hold)
        """
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Calculate the middle band (SMA)
        middle_band = data_copy['Close'].rolling(window=self.window).mean()
        
        # Calculate the standard deviation
        std = data_copy['Close'].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        # Create a new DataFrame with all the necessary columns
        df = pd.DataFrame(index=data_copy.index)
        df['Close'] = data_copy['Close']
        df['middle_band'] = middle_band
        df['upper_band'] = upper_band
        df['lower_band'] = lower_band
        
        # Drop NaN values resulting from the rolling window
        df = df.dropna()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Create boolean series for price below lower band and above upper band
        below_lower = (df['Close'] < df['lower_band']).astype(bool)
        above_upper = (df['Close'] > df['upper_band']).astype(bool)
        
        # Shift to get previous state (avoiding fillna warning)
        was_below_lower = below_lower.shift(1)
        was_above_upper = above_upper.shift(1)
        
        # First value will be NaN, set it to False
        was_below_lower.loc[was_below_lower.index[0]] = False
        was_above_upper.loc[was_above_upper.index[0]] = False
        
        # Generate buy signals: price was below lower band and now is not
        buy_condition = (was_below_lower) & (~below_lower)
        df.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals: price was above upper band and now is not
        sell_condition = (was_above_upper) & (~above_upper)
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def backtest(self, data, initial_capital=10000.0):
        """
        Backtest the Bollinger Bands strategy without shorting.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with 'Close' column
        initial_capital : float, optional
            Initial investment amount (default: 10000.0)
            
        Returns:
        --------
        tuple
            (DataFrame with portfolio value and positions, DataFrame with signals)
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        if signals.empty:
            print("No signals generated. Check your data and parameters.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create a DataFrame for positions and portfolio value
        positions = pd.DataFrame(index=signals.index).copy()
        positions['price'] = signals['Close']
        positions['signal'] = signals['signal']
        
        # Initialize position column
        positions['position'] = 0
        
        # Initialize position_status column for clearer indication of trading activity
        positions['position_status'] = 'OUT OF MARKET'
        
        # Calculate positions (shares held) - No shorting allowed (only 0 or 1)
        current_position = 0
        for i in range(len(positions)):
            signal = positions['signal'].iloc[i]
            
            # Buy signal and not already in a position
            if signal == 1 and current_position == 0:
                current_position = 1
                positions.loc[positions.index[i], 'position'] = current_position
                positions.loc[positions.index[i], 'position_status'] = 'BUY & HOLD'
            # Sell signal and currently in a position
            elif signal == -1 and current_position == 1:
                current_position = 0
                positions.loc[positions.index[i], 'position'] = current_position
                positions.loc[positions.index[i], 'position_status'] = 'SELL'
            # No signal but in a position
            elif current_position == 1:
                positions.loc[positions.index[i], 'position'] = current_position
                positions.loc[positions.index[i], 'position_status'] = 'HOLDING'
            # No signal and not in a position
            else:
                positions.loc[positions.index[i], 'position'] = current_position
                positions.loc[positions.index[i], 'position_status'] = 'OUT OF MARKET'
        
        # Calculate daily returns
        returns = positions['price'].pct_change()
        # Replace NaN values with 0 for the first day
        returns.loc[returns.index[0]] = 0
        positions['returns'] = returns
        
        # Calculate strategy returns (only when we have a position)
        prev_position = positions['position'].shift(1).fillna(0)
        positions['strategy_returns'] = positions['returns'] * prev_position
        
        # Calculate portfolio value
        cumulative_returns = (1 + positions['strategy_returns']).cumprod()
        positions['portfolio_value'] = initial_capital * cumulative_returns
        
        # Calculate number of shares based on initial price
        initial_price = positions['price'].iloc[0]
        share_size = initial_capital / initial_price
        
        # Calculate holdings and cash
        positions['holdings'] = positions['position'] * positions['price'] * share_size
        positions['cash'] = positions['portfolio_value'] - positions['holdings']
        
        # Filter signals for visualization
        signal_points = signals[signals['signal'] != 0].copy()
        
        return positions, signal_points
    
    def optimize(self, data, window_range=(10, 30), num_std_range=(1.5, 2.5), step_size=0.1):
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with 'Close' column
        window_range : tuple, optional
            Range of window sizes to test (default: (10, 30))
        num_std_range : tuple, optional
            Range of standard deviation values to test (default: (1.5, 2.5))
        step_size : float, optional
            Step size for standard deviation values (default: 0.1)
            
        Returns:
        --------
        tuple
            (optimal window size, optimal number of standard deviations, best Sharpe ratio)
        """
        best_sharpe = -np.inf
        best_window = None
        best_num_std = None
        
        # Create ranges for grid search
        window_sizes = range(window_range[0], window_range[1] + 1)
        num_stds = np.arange(num_std_range[0], num_std_range[1] + step_size, step_size)
        
        for window in window_sizes:
            for num_std in num_stds:
                # Update parameters
                self.window = window
                self.num_std = num_std
                
                try:
                    # Run backtest
                    positions, _ = self.backtest(data)
                    
                    if positions.empty:
                        continue
                    
                    # Calculate Sharpe ratio
                    if 'strategy_returns' in positions.columns and not positions['strategy_returns'].empty:
                        returns = positions['strategy_returns'].dropna()
                        if len(returns) > 0 and returns.std() > 0:
                            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                            
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_window = window
                                best_num_std = num_std
                except Exception as e:
                    print(f"Error during optimization with window={window}, num_std={num_std}: {e}")
                    continue
        
        if best_window is None or best_num_std is None:
            print("Optimization failed to find optimal parameters. Using defaults.")
            best_window = 20
            best_num_std = 2.0
            best_sharpe = 0.0
        
        # Reset to optimal parameters
        self.window = best_window
        self.num_std = best_num_std
        
        return best_window, best_num_std, best_sharpe 