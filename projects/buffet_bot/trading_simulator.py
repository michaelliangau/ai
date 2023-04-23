import yfinance as yf
import pandas as pd
import datetime
import IPython
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as tradeapi


class StockSimulator:
    """Stock simulator class for simulating stock trades and calculating profit/loss."""

    def __init__(self, initial_investment, real_trading=False):
        """Initializes the stock simulator.

        Args:
            initial_investment: The initial investment amount.
            real_trading: Whether to use real trading or simulated trading.
        """
        self.stock_data = {}
        self.trades = []
        self.balance = initial_investment
        self.holdings = {}
        self.initial_investment = initial_investment
        self.real_trading = real_trading
        if self.real_trading:
            # Init Alpaca API
            with open("/Users/michael/Desktop/wip/alpaca_credentials.txt", "r") as f:
                ALPACA_API_KEY = f.readline().strip()
                ALPACA_API_SECRET = f.readline().strip()
                BASE_URL = "https://paper-api.alpaca.markets"  # Use the paper trading endpoint for testing
            self.alpaca = tradeapi.REST(
                ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version="v2"
            )

    def is_trading_day(self, ticker, date):
        """Returns True if the stock market is open on the given date."""
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        stock = yf.Ticker(ticker)
        stock_df = stock.history(
            start=date_obj, end=date_obj + datetime.timedelta(days=1)
        )
        return not stock_df.empty

    def find_next_trading_day(self, ticker, start_date):
        """Returns the next trading day for the given ticker and start date."""
        current_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        while not self.is_trading_day(ticker, current_date.strftime("%Y-%m-%d")):
            current_date += datetime.timedelta(days=1)
        return current_date.strftime("%Y-%m-%d")

    def get_stock_data(self, ticker, start_date, end_date):
        """Gets the stock data for the given ticker and date range."""
        stock = yf.Ticker(ticker)
        stock_df = stock.history(start=start_date, end=end_date)
        self.stock_data[ticker] = stock_df

    def get_price_at_time(self, ticker, date):
        """Gets the price of the given ticker at the given date."""
        if date not in self.stock_data[ticker].index:
            # get stock data
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
            end_date_obj = date_obj + relativedelta(months=1)
            end_date = end_date_obj.strftime("%Y-%m-%d")
            self.get_stock_data(ticker, date, end_date)
        return self.stock_data[ticker].loc[date]["Close"]

    def update_holdings(
        self, stocks_dict, date, initial_investment=100000, transaction_cost=0.0001
    ):
        """Updates the holdings based on the given stocks_dict and date.

        Args:
            stocks_dict (dict): A dictionary of stocks and their percentages.
            date (str): The date to update the holdings.
            initial_investment (int, optional): The initial investment. Defaults to 100000.
            transaction_cost (float, optional): The transaction cost as a percentage. Defaults to 0.0001.
        """
        # Calculate the total portfolio value
        total_portfolio_value = 0
        for ticker, percentage in stocks_dict.items():
            if ticker in self.holdings:
                try:
                    current_price = self.get_price_at_time(ticker, date)
                    total_portfolio_value += current_price * self.holdings[ticker]
                except ValueError:
                    print(f"Ticker {ticker} not found. Skipping...")
                    continue

        # If total_portfolio_value is 0, use the initial_investment value
        if total_portfolio_value == 0:
            total_portfolio_value = initial_investment
        else:
            total_portfolio_value += self.balance

        # Update holdings based on the percentage of the total portfolio
        print(f"Updating holdings on {date}...")

        # Sell first
        for ticker, percentage in stocks_dict.items():
            try:
                target_value = total_portfolio_value * (percentage / 100)
                current_price = self.get_price_at_time(ticker, date)
                current_value = current_price * self.holdings.get(ticker, 0)
                target_shares = target_value / current_price

                if target_value < current_value:
                    shares_to_sell = self.holdings.get(ticker, 0) - target_shares
                    trade_value = current_price * shares_to_sell
                    trade_cost = trade_value * transaction_cost
                    self.sell(ticker, date, shares_to_sell)
                    self.balance += trade_cost
            except ValueError:
                print(f"Ticker {ticker} not found. Skipping sell...")
                continue

        # Buy second
        for ticker, percentage in stocks_dict.items():
            try:
                target_value = total_portfolio_value * (percentage / 100)
                current_price = self.get_price_at_time(ticker, date)
                current_value = current_price * self.holdings.get(ticker, 0)
                target_shares = target_value / current_price

                if target_value > current_value:
                    shares_to_buy = target_shares - self.holdings.get(ticker, 0)
                    trade_value = current_price * shares_to_buy
                    trade_cost = trade_value * transaction_cost
                    if self.balance - trade_value - trade_cost >= 0:
                        self.buy(ticker, date, shares_to_buy)
                        self.balance -= trade_value + trade_cost
                    else:
                        print(
                            f"Not enough cash to buy {shares_to_buy} shares of {ticker} at {current_price} on {date}."
                        )
            except ValueError:
                print(f"Ticker {ticker} not found. Skipping buy...")
                continue

    def buy(self, ticker, date, shares):
        shares = int(shares)
        if self.real_trading:
            try:
                self.alpaca.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side="buy",
                    type="market",
                    time_in_force="gtc",
                )
                print(f"Bought {shares} shares of {ticker}.")
            except Exception as e:
                print(f"Error buying {shares} shares of {ticker}: {e}")
        else:
            # Simulated trading logic
            price = self.get_price_at_time(ticker, date)
            trade_value = price * shares
            if self.balance - trade_value >= 0:
                self.balance -= trade_value
                self.trades.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "shares": shares,
                        "price": price,
                        "trade_value": trade_value,
                        "action": "buy",
                    }
                )
                self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
            else:
                print(
                    f"Not enough cash to buy {shares} shares of {ticker} at {price} on {date}."
                )

    def sell(self, ticker, date, shares):
        shares = int(shares)
        if self.real_trading:
            try:
                self.alpaca.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                print(f"Sold {shares} shares of {ticker}.")
            except Exception as e:
                print(f"Error selling {shares} shares of {ticker}: {e}")
        else:
            # Simulated trading logic
            price = self.get_price_at_time(ticker, date)
            trade_value = price * shares
            if self.holdings.get(ticker, 0) >= shares:
                self.balance += trade_value
                self.trades.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "shares": -shares,
                        "price": price,
                        "trade_value": -trade_value,
                        "action": "sell",
                    }
                )
                self.holdings[ticker] = self.holdings.get(ticker, 0) - shares
            else:
                print(f"Not enough shares of {ticker} to sell {shares} on {date}.")

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
                    "shares": total_shares,
                    "price": current_price,
                    "position_value": position_value,
                }
            except Exception as e:
                print(f"Error getting portfolio position for {ticker} on {date}: {e}")

        portfolio_position["total_portfolio_value"] = portfolio_value
        portfolio_position["cash_balance"] = self.balance
        total_value = self.balance + portfolio_value
        portfolio_position["total_value"] = total_value
        # Store the date in the desired format
        portfolio_position["date"] = date
        return portfolio_position

    def reset(self):
        """Reset the simulator's state back to the initial state."""
        self.stock_data = {}
        self.trades = []
        self.balance = self.initial_investment
        self.holdings = {}
