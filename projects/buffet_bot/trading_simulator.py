import yfinance as yf
import pandas as pd
import datetime
import IPython

class StockSimulator:
    """Stock simulator class for simulating stock trades and calculating profit/loss.
    
    Example usage:
    
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
    """
    def __init__(self):
        self.stock_data = {}
        self.trades = []
        self.balance = 0
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
       
    def update_holdings(self, stocks_dict, date):
        # Calculate the total portfolio value
        total_portfolio_value = 0
        for ticker, percentage in stocks_dict.items():
            if ticker in self.holdings:
                current_price = self.get_price_at_time(ticker, date)
                total_portfolio_value += current_price * self.holdings[ticker]

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
        for ticker in self.holdings:
            current_price = self.get_price_at_time(ticker, date)
            total_shares = self.holdings[ticker]
            position_value = total_shares * current_price
            portfolio_position.append(f"Current position for {ticker}: {total_shares} shares at a market price of ${current_price:.2f} per share. Total position value: ${position_value:.2f}.")
        return '\n'.join(portfolio_position)

