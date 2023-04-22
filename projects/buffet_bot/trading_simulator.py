import yfinance as yf
import pandas as pd
import datetime
import IPython
from dateutil.relativedelta import relativedelta

class StockSimulator:
    """Stock simulator class for simulating stock trades and calculating profit/loss.
    """
    def __init__(self, initial_investment):
        self.stock_data = {}
        self.trades = []
        self.balance = initial_investment
        self.holdings = {}
        self.initial_investment = initial_investment
    
    def is_trading_day(self, ticker, date):
        """Returns True if the stock market is open on the given date."""
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        stock = yf.Ticker(ticker)
        stock_df = stock.history(start=date_obj, end=date_obj + datetime.timedelta(days=1))
        return not stock_df.empty

    def find_next_trading_day(self, ticker, start_date):
        """Returns the next trading day for the given ticker and start date."""
        current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        while not self.is_trading_day(ticker, current_date.strftime('%Y-%m-%d')):
            current_date += datetime.timedelta(days=1)
        return current_date.strftime('%Y-%m-%d')
       
    def update_holdings(self, stocks_dict, date, initial_investment=100000):
        """Updates the holdings based on the given stocks_dict and date.
        
        Args:
            stocks_dict (dict): A dictionary of stocks and their percentages.
            date (str): The date to update the holdings.
            initial_investment (int, optional): The initial investment. Defaults to 100000.
        """
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
        print(f"Updating holdings on {date}...")
        for ticker, percentage in stocks_dict.items():
                target_value = total_portfolio_value * (percentage / 100)
                current_price = self.get_price_at_time(ticker, date)
                current_value = current_price * self.holdings.get(ticker, 0)
                target_shares = target_value / current_price

                if target_value > current_value:
                    shares_to_buy = target_shares - self.holdings.get(ticker, 0)
                    self.buy(ticker, date, shares_to_buy)
                    print(f"Bought {shares_to_buy} shares of {ticker}.")
                elif target_value < current_value:
                    shares_to_sell = self.holdings.get(ticker, 0) - target_shares
                    self.sell(ticker, date, shares_to_sell)
                    print(f"Sold {shares_to_sell} shares of {ticker}.")
                else:
                    print(f"No change in holdings for {ticker}.")

    def get_stock_data(self, ticker, start_date, end_date):
        """Gets the stock data for the given ticker and date range."""
        stock = yf.Ticker(ticker)
        stock_df = stock.history(start=start_date, end=end_date)
        self.stock_data[ticker] = stock_df

    def get_price_at_time(self, ticker, date):
        """Gets the price of the given ticker at the given date."""
        if date not in self.stock_data[ticker].index:
            # get stock data
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
            end_date_obj = date_obj + relativedelta(months=1)
            end_date = end_date_obj.strftime('%Y-%m-%d')
            self.get_stock_data(ticker, date, end_date)

        return self.stock_data[ticker].loc[date]['Adj Close']

    def buy(self, ticker, date, shares):
        """Buys the given number of shares of the given ticker at the given date."""
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
        """Sells the given number of shares of the given ticker at the given date."""
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

    def get_portfolio_position(self, date):
        """Returns the portfolio position for the given date."""
        portfolio_position = {}
        portfolio_value = 0

        for ticker in self.holdings:
            try:
                current_price = self.get_price_at_time(ticker, date)
                total_shares = self.holdings[ticker]
                position_value = total_shares * current_price
                portfolio_value += position_value
                portfolio_position[ticker] = {
                    'shares': total_shares,
                    'price': current_price,
                    'position_value': position_value
                }
            except Exception as e:
                print(f"Error getting portfolio position for {ticker} on {date}: {e}")

        portfolio_position['total_portfolio_value'] = portfolio_value
        portfolio_position['cash_balance'] = self.balance
        total_value = self.balance + portfolio_value
        portfolio_position['total_value'] = total_value
        # Store the date in the desired format
        portfolio_position['date'] = date
        return portfolio_position


    def reset(self):
        """Reset the simulator's state back to the initial state."""
        self.stock_data = {}
        self.trades = []
        self.balance = self.initial_investment
        self.holdings = {}

