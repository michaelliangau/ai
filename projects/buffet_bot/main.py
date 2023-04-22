# Native imports
import json
import datetime

# Our imports
from llm import BuffetBot
from trading_simulator import StockSimulator

# Third party imports
import IPython
from dateutil.relativedelta import relativedelta


def main():
    # Vars
    investor_type = 'value'
    initial_investment = 100_000
    context_window_date = '2018-01-01'
    investment_schedule = 'monthly'


    # Init bots and simulator
    bot = BuffetBot(llm="anthropic", vector_context=False)
    simulator = StockSimulator(initial_investment)
    for _ in range(1):
        current_holdings = simulator.holdings

        # Find next trading day if current date is not a trading day
        while not simulator.is_trading_day(ticker='MSFT', date=context_window_date):
            context_window_date = simulator.find_next_trading_day(ticker='MSFT', start_date=context_window_date)

        if investor_type == 'value':
            llm_prompt = f'You are Warren Buffett, one of the world\'s most successful value investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Warren Buffett, you have a portfolio to invest in any investment vehicle. Build your portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding in JSON format. Do not return any other text.'

        response = bot.get_response(llm_prompt)
        updated_portfolio = json.loads(response['completion'])

        # Update current holdings in stock simulator
        updated_portfolio = {key.replace(".", "-"): value for key, value in updated_portfolio.items()}
        for ticker in updated_portfolio.keys():
            # Replace tickers . with -
            ticker = ticker.replace('.', '-')

            context_window_date_obj = datetime.datetime.strptime(context_window_date, '%Y-%m-%d')
            end_date_obj = context_window_date_obj + relativedelta(months=1)
            end_date = end_date_obj.strftime('%Y-%m-%d')
            simulator.get_stock_data(ticker, start_date=context_window_date, end_date=end_date)
        print(updated_portfolio)

        simulator.update_holdings(updated_portfolio, context_window_date, initial_investment)

        # Print current portfolio position
        print(simulator.get_portfolio_position(context_window_date))

        if investment_schedule == 'monthly':
            # Increment context_start_date by 1 month
            context_window_date_obj = datetime.datetime.strptime(context_window_date, '%Y-%m-%d')
            context_window_date_obj = context_window_date_obj + relativedelta(months=1)
            context_window_date = context_window_date_obj.strftime('%Y-%m-%d')


if __name__ == '__main__':
    main()