import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StrategyBacktester:
    def __init__(self, ticker, start_date, end_date, initial_capital=10000):
        """
        Initialize the backtester
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            initial_capital: Starting cash amount
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.positions = []
        self.dividend_payments = []
        self.cash = initial_capital
        self.holdings = 0
        
    def fetch_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        
        # Download stock data with actions (dividends and splits)
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, 
                          progress=False, actions=True)
        
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        self.data = data
        
        # Add dividends column if not present (some stocks don't pay dividends)
        if 'Dividends' not in self.data.columns:
            self.data['Dividends'] = 0.0
        else:
            # Fill NaN dividends with 0
            self.data['Dividends'] = self.data['Dividends'].fillna(0.0)
        
        self.data['Returns'] = self.data['Close'].pct_change()
        
        total_dividends = self.data['Dividends'].sum()
        print(f"Downloaded {len(self.data)} days of data")
        print(f"Total dividends in period: ${total_dividends:.2f} per share")
        
        return self.data
    
    def moving_average_crossover(self, short_window=20, long_window=50):
        """
        Simple Moving Average Crossover Strategy
        Buy when short MA crosses above long MA
        Sell when short MA crosses below long MA
        """
        self.data['SMA_short'] = self.data['Close'].rolling(window=short_window).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['SMA_short'] > self.data['SMA_long'], 'Signal'] = 1
        self.data['Position'] = self.data['Signal'].diff()
        
        return self.data
    
    def rsi_strategy(self, period=14, oversold=30, overbought=70):
        """
        RSI Strategy
        Buy when RSI < oversold threshold
        Sell when RSI > overbought threshold
        """
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['RSI'] < oversold, 'Signal'] = 1  # Buy
        self.data.loc[self.data['RSI'] > overbought, 'Signal'] = -1  # Sell
        self.data['Position'] = self.data['Signal'].diff()
        
        return self.data
    
    def backtest(self):
        """Execute backtest based on generated signals"""
        self.cash = self.initial_capital
        self.holdings = 0
        portfolio_value = []
        total_dividends_received = 0
        
        # Convert to numpy arrays for faster access
        positions = self.data['Position'].values
        closes = self.data['Close'].values
        dividends = self.data['Dividends'].values
        dates = self.data.index
        
        for idx in range(len(self.data)):
            # Get position value, handling NaN
            position = positions[idx]
            if np.isnan(position):
                position = 0
            
            # Get current price and dividend
            current_price = closes[idx]
            dividend_per_share = dividends[idx]
            
            # Collect dividends if we own shares
            if self.holdings > 0 and dividend_per_share > 0:
                dividend_payment = self.holdings * dividend_per_share
                self.cash += dividend_payment
                total_dividends_received += dividend_payment
                
                # Record dividend payment
                self.dividend_payments.append({
                    'Date': dates[idx],
                    'Shares_Held': self.holdings,
                    'Dividend_Per_Share': dividend_per_share,
                    'Total_Payment': dividend_payment
                })
            
            # Buy signal
            if position == 1 and self.cash > 0:
                shares_to_buy = int(self.cash / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.holdings += shares_to_buy
                    self.cash -= cost
                    self.positions.append({
                        'Date': dates[idx],
                        'Action': 'BUY',
                        'Shares': shares_to_buy,
                        'Price': current_price,
                        'Cost': cost
                    })
            
            # Sell signal
            elif position == -1 and self.holdings > 0:
                proceeds = self.holdings * current_price
                self.positions.append({
                    'Date': dates[idx],
                    'Action': 'SELL',
                    'Shares': self.holdings,
                    'Price': current_price,
                    'Proceeds': proceeds
                })
                self.cash += proceeds
                self.holdings = 0
            
            # Track portfolio value
            total_value = self.cash + (self.holdings * current_price)
            portfolio_value.append(total_value)
        
        self.data['Portfolio_Value'] = portfolio_value
        self.total_dividends = total_dividends_received
        
        print(f"\nTotal dividends received during backtest: ${total_dividends_received:.2f}")
        
        return self.data
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        # Get final portfolio value
        final_value = self.data['Portfolio_Value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Buy and hold comparison (with dividends)
        first_close = float(self.data['Close'].iloc[0])
        last_close = float(self.data['Close'].iloc[-1])
        buy_hold_shares = self.initial_capital / first_close
        
        # Calculate dividends for buy-and-hold strategy
        total_dividends_per_share = self.data['Dividends'].sum()
        buy_hold_dividends = buy_hold_shares * total_dividends_per_share
        
        buy_hold_value = (buy_hold_shares * last_close) + buy_hold_dividends
        buy_hold_return = (buy_hold_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate daily returns - ensure we have numeric values
        portfolio_values = pd.Series(self.data['Portfolio_Value'].values.flatten())
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
        
        # Win rate
        num_trades = len([p for p in self.positions if p['Action'] == 'SELL'])
        
        # Calculate dividends received
        dividends_received = getattr(self, 'total_dividends', 0)
        
        metrics = {
            'Initial Capital': f"${self.initial_capital:,.2f}",
            'Final Value': f"${final_value:,.2f}",
            'Total Dividends Received': f"${dividends_received:.2f}",
            'Strategy Return': f"{total_return:.2f}%",
            'Buy & Hold Return': f"{buy_hold_return:.2f}%",
            'Buy & Hold Dividends': f"${buy_hold_dividends:.2f}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Number of Trades': num_trades
        }
        
        return metrics
    
    def print_results(self):
        """Print backtest results"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print(f"BACKTEST RESULTS FOR {self.ticker}")
        print("="*50)
        
        for key, value in metrics.items():
            print(f"{key:.<30} {value}")
        
        # Print dividend payments
        if self.dividend_payments:
            print("\n" + "="*50)
            print("DIVIDEND PAYMENTS RECEIVED")
            print("="*50)
            print(f"{'Date':<12} {'Shares':<10} {'Per Share':<12} {'Total Payment':<15}")
            print("-" * 50)
            
            for div in self.dividend_payments:
                print(f"{div['Date'].strftime('%Y-%m-%d'):<12} "
                      f"{div['Shares_Held']:<10} "
                      f"${div['Dividend_Per_Share']:<11.4f} "
                      f"${div['Total_Payment']:<14.2f}")
            
            total_div = sum(d['Total_Payment'] for d in self.dividend_payments)
            print("-" * 50)
            print(f"{'TOTAL DIVIDENDS:':<35} ${total_div:.2f}")
        else:
            print("\n" + "="*50)
            print("DIVIDEND PAYMENTS RECEIVED")
            print("="*50)
            print("No dividends received (stock may not pay dividends or no shares held during dividend dates)")
        
        print("\n" + "="*50)
        print("TRADE HISTORY")
        print("="*50)
        
        for pos in self.positions[:10]:  # Show first 10 trades
            print(f"{pos['Date'].strftime('%Y-%m-%d')} | {pos['Action']:4} | "
                  f"Shares: {pos['Shares']:6} | Price: ${pos['Price']:.2f}")
        
        if len(self.positions) > 10:
            print(f"... and {len(self.positions) - 10} more trades")


# Example usage
if __name__ == "__main__":
    # Set parameters
    # ticker = "AAPL"
    ticker = "1155.KL" # maybank ß
    # ticker = "5123.KL" # sentral Reit
    start_date = "2020-01-01"
    end_date = "2025-11-01"
    
    # Create backtester
    bt = StrategyBacktester(ticker, start_date, end_date, initial_capital=10000)
    
    # Fetch data
    bt.fetch_data()
    
    # Choose strategy (uncomment one):
    bt.moving_average_crossover(short_window=20, long_window=50)
    # bt.rsi_strategy(period=14, oversold=30, overbought=70)
    
    # Run backtest
    bt.backtest()
    
    # Print results
    bt.print_results()