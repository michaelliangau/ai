# Native imports
import json
import datetime

# Our imports
from llm import BuffetBot
from trading_simulator import StockSimulator

# Third party imports
import IPython
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def find_next_trading_day(context_window_date, simulator, ticker='MSFT'):
    """Finds the next trading day for the given ticker after the given date."""
    while not simulator.is_trading_day(ticker, date=context_window_date):
        context_window_date = simulator.find_next_trading_day(ticker, start_date=context_window_date)
    return context_window_date

def get_llm_response(bot, investor_type, context_window_date, current_holdings):
    """Gets the response from the LLM and returns the updated portfolio."""
    if investor_type == 'value':
        llm_prompt = f'You are Warren Buffett, one of the world\'s most successful value investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Warren Buffett, you have a portfolio to invest in any investment vehicle. Build your portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding in JSON format. Do not return any other text.'

    response = bot.get_response(llm_prompt)
    updated_portfolio = json.loads(response['completion'])

    return updated_portfolio

def update_holdings(simulator, updated_portfolio, context_window_date, initial_investment, prev_updated_portfolio):
    """Updates the holdings in the simulator based on the updated portfolio."""
    updated_portfolio = {key.replace(".", "-"): value for key, value in updated_portfolio.items()}

    for ticker in updated_portfolio.keys():
        context_window_date_obj = datetime.datetime.strptime(context_window_date, '%Y-%m-%d')
        end_date_obj = context_window_date_obj + relativedelta(months=1)
        end_date = end_date_obj.strftime('%Y-%m-%d')
        simulator.get_stock_data(ticker, start_date=context_window_date, end_date=end_date)

    if updated_portfolio != prev_updated_portfolio:
        simulator.update_holdings(updated_portfolio, context_window_date, initial_investment)

    # Update prev_updated_portfolio
    prev_updated_portfolio = updated_portfolio

    return prev_updated_portfolio

def increment_time(investment_schedule, context_window_date):
    """Increments the context window date based on the investment schedule."""
    if investment_schedule == 'monthly':
        # Increment context_start_date by 1 month
        context_window_date_obj = datetime.datetime.strptime(context_window_date, '%Y-%m-%d')
        context_window_date_obj = context_window_date_obj + relativedelta(months=1)
        context_window_date = context_window_date_obj.strftime('%Y-%m-%d')
    return context_window_date

def main():
    # Vars
    investor_type = 'value'
    initial_investment = 100_000
    context_window_date = '2018-01-01'
    investment_schedule = 'monthly'
    num_simulated_months = 12
    num_simulations = 10

    # Init bots and simulator
    bot = BuffetBot(llm="anthropic", vector_context=False)
    simulator = StockSimulator(initial_investment)

    # Run simulation
    for sim in range(num_simulations):

        results = []
        prev_updated_portfolio = None

        for _ in range(num_simulated_months):
            try:
                current_holdings = simulator.holdings

                # Find next trading day
                context_window_date = find_next_trading_day(context_window_date, simulator)

                # Get response from LLM
                updated_portfolio = get_llm_response(bot, investor_type, context_window_date, current_holdings)
                print(updated_portfolio)

                # If updated_portfolio is different from the previous one, update holdings
                prev_updated_portfolio = update_holdings(simulator, updated_portfolio, context_window_date, initial_investment, prev_updated_portfolio)

                # Print current portfolio position
                portfolio_position = simulator.get_portfolio_position(context_window_date)
                results.append(portfolio_position)
                print(portfolio_position)
                print(f'Current portfolio value at {context_window_date}: {portfolio_position["total_value"]}')


                # Increment time
                context_window_date = increment_time(investment_schedule, context_window_date)

            except Exception as e:
                print(e)

        # Save the results
        with open(f'output/sim_{sim}_results.json', 'w') as f:
            json.dump(results, f)

        # Reset the context_window_date and simulator for the next simulation
        context_window_date = '2018-01-01'
        simulator.reset()            

if __name__ == '__main__':
    main()
