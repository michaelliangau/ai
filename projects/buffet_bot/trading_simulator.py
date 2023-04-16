import yfinance as yf
import pandas as pd
import datetime

class StockSimulator:
    def __init__(self):
        self.stock_data = {}
        self.trades = []
        self.balance = 0
        self.holdings = {}

    def get_stock_data(self, ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        stock_df = stock.history(start=start_date, end=end_date)
        self.stock_data[ticker] = stock_df

    def get_price_at_time(self, ticker, date):
        return self.stock_data[ticker].loc[date]['Close']

    def buy(self, ticker, date, shares):
        price = self.get_price_at_time(ticker, date)
        trade_value = price * shares
        self.balance -= trade_value
        self.trades.append({
            'ticker': ticker,
            'date': date,
            'shares': shares,
            'price': price,
            'trade_value': trade_value,
            'action': 'buy'
        })
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares

    def sell(self, ticker, date, shares):
        price = self.get_price_at_time(ticker, date)
        trade_value = price * shares
        self.balance += trade_value
        self.trades.append({
            'ticker': ticker,
            'date': date,
            'shares': -shares,
            'price': price,
            'trade_value': -trade_value,
            'action': 'sell'
        })
        self.holdings[ticker] = self.holdings.get(ticker, 0) - shares

    def profit_loss(self, ticker, current_date):
        current_price = self.get_price_at_time(ticker, current_date)
        initial_investment = sum([trade['trade_value'] for trade in self.trades if trade['ticker'] == ticker])
        return self.holdings[ticker] * current_price - initial_investment

    def get_portfolio_position(self, date):
        portfolio_position = []
        for ticker in self.holdings:
            current_price = self.get_price_at_time(ticker, date)
            total_shares = self.holdings[ticker]
            position_value = total_shares * current_price
            portfolio_position.append(f"Current position for {ticker}: {total_shares} shares at a market price of ${current_price:.2f} per share. Total position value: ${position_value:.2f}.")
        return '\n'.join(portfolio_position)


# Create a StockSimulator instance
simulator = StockSimulator()

# Fetch stock data for Apple and Microsoft
simulator.get_stock_data('AAPL', '2020-01-01', '2021-01-01')
simulator.get_stock_data('MSFT', '2020-01-01', '2021-01-01')

# Get and print stock prices at specific dates
apple_price = simulator.get_price_at_time('AAPL', '2020-02-03')
microsoft_price = simulator.get_price_at_time('MSFT', '2020-02-03')
print(f"AAPL price on 2020-02-03: ${apple_price:.2f}")
print(f"MSFT price on 2020-02-03: ${microsoft_price:.2f}\n")

# Buy shares
simulator.buy('AAPL', '2020-02-03', 10)
simulator.buy('MSFT', '2020-02-03', 20)

# Sell shares
simulator.sell('AAPL', '2020-02-28', 5)
simulator.sell('MSFT', '2020-02-28', 10)

# Calculate and print profit/loss
apple_profit_loss = simulator.profit_loss('AAPL', '2020-02-28')
microsoft_profit_loss = simulator.profit_loss('MSFT', '2020-02-28')
print(f"AAPL profit/loss on 2020-02-28: ${apple_profit_loss:.2f}")
print(f"MSFT profit/loss on 2020-02-28: ${microsoft_profit_loss:.2f}\n")

# Get and print portfolio position
portfolio_position = simulator.get_portfolio_position('2020-02-28')
print("Portfolio position on 2020-02-28:")
print(portfolio_position)
