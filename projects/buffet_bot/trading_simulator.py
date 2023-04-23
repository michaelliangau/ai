# Our imports
import utils

# Third party imports
import yfinance as yf
import datetime
import IPython
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

    def get_price_at_time(self, ticker: str, date: str) -> float:
        """Returns the price of the given ticker at the given date."""
        if self.real_trading:
            # Use Alpaca API for real trading
            latest_trade = self.alpaca.get_latest_trade(ticker)
            return latest_trade.p
        else:
            # Use the original method for simulated trading
            return self.stock_data[ticker].loc[date].Close

    def _calculate_target_shares(
        self, ticker, date, stocks_dict, total_portfolio_value
    ):
        """Calculates the target shares for the given ticker."""
        percentage = stocks_dict[ticker]
        target_value = total_portfolio_value * (percentage / 100)
        current_price = self.get_price_at_time(ticker, date)
        current_value = current_price * self.holdings.get(ticker, 0)
        target_shares = target_value / current_price
        return target_shares, current_value, current_price, target_value

    def _execute_trade(
        self, action, ticker, date, shares, current_price, transaction_cost
    ):
        """Executes a trade.

        Args:
            action: The action to take. Must be "buy" or "sell".
            ticker: The ticker of the stock to trade.
            date: The date of the trade.
            shares: The number of shares to trade.
            current_price: The current price of the stock.
            transaction_cost: The transaction cost of the trade.
        """
        if action not in ["buy", "sell"]:
            raise ValueError("Invalid action. Must be 'buy' or 'sell'.")

        shares = int(shares)
        trade_value = current_price * shares
        cost = trade_value * transaction_cost

        if self.real_trading:
            try:
                order = self.alpaca.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side=action,
                    type="market",
                    time_in_force="gtc",
                )
                print(f"{action.capitalize()} {shares} shares of {ticker}.")
            except Exception as e:
                print(f"Error {action}ing {shares} shares of {ticker}: {e}")
                return

            # Get order status
            order_status = self.alpaca.get_order(order.id).status

            if order_status == "accepted":
                # Update holdings, balance, and trades for real trading
                if action == "buy":
                    if self.balance - trade_value - cost >= 0:
                        self.balance -= trade_value + cost
                elif action == "sell":
                    if self.holdings.get(ticker, 0) >= shares:
                        self.balance += trade_value - cost

                self.trades.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "shares": shares if action == "buy" else -shares,
                        "price": current_price,
                        "trade_value": trade_value if action == "buy" else -trade_value,
                        "action": action,
                        "cost": cost,
                    }
                )
                self.holdings[ticker] = self.holdings.get(ticker, 0) + (
                    shares if action == "buy" else -shares
                )

        else:
            # Simulated trading logic
            if action == "buy":
                max_shares_to_buy = int((self.balance / (1 + transaction_cost)) / current_price)
                if max_shares_to_buy < shares:
                    shares = max_shares_to_buy

                if self.balance - (shares * current_price) * (1 + transaction_cost) >= 0:
                    self.balance -= (shares * current_price) * (1 + transaction_cost)
                else:
                    print(
                        f"Not enough cash to buy {shares} shares of {ticker} at {current_price} on {date}."
                    )
                    return
            elif action == "sell":
                if self.holdings.get(ticker, 0) >= shares:
                    self.balance += trade_value - cost
                else:
                    print(f"Not enough shares of {ticker} to sell {shares} on {date}.")
                    return

            self.trades.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "shares": shares if action == "buy" else -shares,
                    "price": current_price,
                    "trade_value": trade_value if action == "buy" else -trade_value,
                    "action": action,
                    "cost": cost,
                }
            )
            self.holdings[ticker] = self.holdings.get(ticker, 0) + (
                shares if action == "buy" else -shares
            )

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
        # Get stock data
        if not self.real_trading:
            end_date = utils.add_one_month(date)
            for ticker in stocks_dict.keys():
                self.get_stock_data(ticker, start_date=date, end_date=end_date)

        # Calculate the total portfolio value
        total_portfolio_value = 0
        for ticker, _ in stocks_dict.items():
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
        for ticker, _ in stocks_dict.items():
            try:
                (
                    target_shares,
                    current_value,
                    current_price,
                    target_value,
                ) = self._calculate_target_shares(
                    ticker, date, stocks_dict, total_portfolio_value
                )
                if current_value > target_value:
                    shares_to_sell = self.holdings.get(ticker, 0) - target_shares
                    self._execute_trade(
                        "sell",
                        ticker,
                        date,
                        shares_to_sell,
                        current_price,
                        transaction_cost,
                    )
            except ValueError:
                print(f"Ticker {ticker} not found. Skipping sell...")
                continue

        # Buy second
        for ticker, _ in stocks_dict.items():
            try:
                (
                    target_shares,
                    current_value,
                    current_price,
                    target_value,
                ) = self._calculate_target_shares(
                    ticker, date, stocks_dict, total_portfolio_value
                )
                if current_value < target_value:
                    shares_to_buy = target_shares - self.holdings.get(ticker, 0)
                    self._execute_trade(
                        "buy",
                        ticker,
                        date,
                        shares_to_buy,
                        current_price,
                        transaction_cost,
                    )
            except ValueError:
                print(f"Ticker {ticker} not found. Skipping buy...")
                continue

    def get_portfolio_position(self, date):
        """Returns the portfolio position for the given date."""
        portfolio_position = {}
        portfolio_value = 0

        for ticker in self.holdings:
            try:
                end_date = utils.add_one_month(date)
                self.get_stock_data(ticker, start_date=date, end_date=end_date)
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
