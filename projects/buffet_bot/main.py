# Native imports
import json

# Our imports
import sys

sys.path.append("../..")
import common.utils as common_utils
from llm import BuffetBot
from trading_simulator import StockSimulator
import utils

# Third party imports
import IPython
import traceback


def main():
    # Vars
    investor_type = "value"
    initial_investment = 100_000
    context_window_date = "2018-01-01"
    investment_schedule = "monthly"
    num_simulated_months = 48
    num_simulations = 2
    llm_additional_context = "news"
    experiment_folder_path = "output/experiments/news_context_ss_200_filtered"
    additional_context_dataset_path = "context_data/huff_news_with_impact_scores.json"
    additional_context_sample_size = (
        200  # Only used if llm_additional_context == "news"
    )
    transaction_cost = 0.0001  # TODO implement

    # Creates output folder if it doesn't exist
    common_utils.create_folder(experiment_folder_path)

    # Init bots and simulator
    if llm_additional_context == "news":
        bot = BuffetBot(
            llm="anthropic",
            additional_context=llm_additional_context,
            additional_context_sample_size=additional_context_sample_size,
            additional_context_dataset_path=additional_context_dataset_path
        )
    else:
        bot = BuffetBot(llm="anthropic", additional_context=llm_additional_context)
    simulator = StockSimulator(initial_investment)

    # Run simulation
    for sim in range(num_simulations):
        results = []
        prev_updated_portfolio = None

        for _ in range(num_simulated_months):
            try:
                current_holdings = simulator.holdings

                # Find next trading day
                context_window_date = utils.find_next_trading_day(
                    context_window_date, simulator
                )

                # Get response from LLM
                updated_portfolio = utils.get_llm_response(
                    bot, investor_type, context_window_date, current_holdings
                )
                print(updated_portfolio)

                # If updated_portfolio is different from the previous one, update holdings
                prev_updated_portfolio = utils.update_holdings(
                    simulator,
                    updated_portfolio,
                    context_window_date,
                    initial_investment,
                    prev_updated_portfolio,
                )

                # Print current portfolio position
                portfolio_position = simulator.get_portfolio_position(
                    context_window_date
                )
                results.append(portfolio_position)
                print(portfolio_position)
                print(
                    f'Current portfolio value at {context_window_date}: {portfolio_position["total_value"]}'
                )

                # Increment time
                context_window_date = utils.increment_time(
                    investment_schedule, context_window_date
                )

            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()  # This line prints the full stack trac
                IPython.embed()

        # Save the results
        with open(f"{experiment_folder_path}/sim_{sim}.json", "w") as f:
            json.dump(results, f)

        # Reset the context_window_date and simulator for the next simulation
        context_window_date = "2018-01-01"
        simulator.reset()


if __name__ == "__main__":
    main()
