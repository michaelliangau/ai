# Native imports
import json

# Our imports
from llm import BuffetBot
from trading_simulator import StockSimulator
import utils

# Third party imports
import IPython

def main():
    # Vars
    investor_type = 'value'
    initial_investment = 100_000
    context_window_date = '2018-01-01'
    investment_schedule = 'monthly'
    num_simulated_months = 48
    num_simulations = 3

    # Init bots and simulator
    bot = BuffetBot(llm="anthropic", vector_context=False, store_conversation_history=False)
    simulator = StockSimulator(initial_investment)

    # Run simulation
    for sim in range(num_simulations):
        results = []
        prev_updated_portfolio = None

        for _ in range(num_simulated_months):
            try:
                current_holdings = simulator.holdings

                # Find next trading day
                context_window_date = utils.find_next_trading_day(context_window_date, simulator)

                # Get response from LLM
                updated_portfolio = utils.get_llm_response(bot, investor_type, context_window_date, current_holdings)
                print(updated_portfolio)

                # If updated_portfolio is different from the previous one, update holdings
                prev_updated_portfolio = utils.update_holdings(simulator, updated_portfolio, context_window_date, initial_investment, prev_updated_portfolio)

                # Print current portfolio position
                portfolio_position = simulator.get_portfolio_position(context_window_date)
                results.append(portfolio_position)
                print(portfolio_position)
                print(f'Current portfolio value at {context_window_date}: {portfolio_position["total_value"]}')

                # Increment time
                context_window_date = utils.increment_time(investment_schedule, context_window_date)

            except Exception as e:
                print(e)

        # Save the results
        with open(f'output/sim_{sim}.json', 'w') as f:
            json.dump(results, f)

        # Reset the context_window_date and simulator for the next simulation
        context_window_date = '2018-01-01'
        simulator.reset()            

if __name__ == '__main__':
    main()
