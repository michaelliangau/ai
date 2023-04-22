import yfinance as yf
import pandas as pd
import datetime
import IPython

class StockSimulator:
    """Stock simulator class for simulating stock trades and calculating profit/loss.
    """
    def __init__(self, initial_investment):
        self.stock_data = {}
        self.trades = []
        self.balance = initial_investment
        self.holdings = {}
    
    def is_trading_day(self, ticker, date):
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        stock = yf.Ticker(ticker)
        stock_df = stock.history(start=date_obj, end=date_obj + datetime.timedelta(days=1))
        return not stock_df.empty


    def find_next_trading_day(self, ticker, start_date):
        current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        while not self.is_trading_day(ticker, current_date.strftime('%Y-%m-%d')):
            current_date += datetime.timedelta(days=1)
        return current_date.strftime('%Y-%m-%d')
       
    def update_holdings(self, stocks_dict, date, initial_investment=100000):
        # Calculate the total portfolio value
        total_portfolio_value = 0
        for ticker, percentage in stocks_dict.items():
            if ticker in self.holdings:
                current_price = self.get_price_at_time(ticker, date)
                total_portfolio_value += current_price * self.holdings[ticker]

        # If total_portfolio_value is 0, use the initial_investment value
        if total_portfolio_value == 0:
            total_portfolio_value = initial_investment

        # Update holdings based on the percentage of the total portfolio
        for ticker, percentage in stocks_dict.items():
            target_value = total_portfolio_value * (percentage / 100)
            current_price = self.get_price_at_time(ticker, date)
            current_value = current_price * self.holdings.get(ticker, 0)
            target_shares = int(target_value / current_price)

            if target_value > current_value:
                self.buy(ticker, date, target_shares - self.holdings.get(ticker, 0))
            elif target_value < current_value:
                self.sell(ticker, date, self.holdings.get(ticker, 0) - target_shares)
            else:
                continue

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
        portfolio_value = 0
        for ticker in self.holdings:
            current_price = self.get_price_at_time(ticker, date)
            total_shares = self.holdings[ticker]
            position_value = total_shares * current_price
            portfolio_value += position_value
            portfolio_position.append(f"Current position for {ticker}: {total_shares} shares at a market price of ${current_price:.2f} per share. Total position value: ${position_value:.2f}.")
        portfolio_position.append(f"Total portfolio value: ${portfolio_value:.2f}.")
        portfolio_position.append(f"Current cash balance: ${self.balance:.2f}.")
        return '\n'.join(portfolio_position)


