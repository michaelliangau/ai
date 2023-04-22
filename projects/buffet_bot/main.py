# Native imports
import json
import argparse

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
import importlib.util

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/growth_config.py",
                    help="Path to the configuration file.")
args = parser.parse_args()


def main(config_path: str):
    # Init configs
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.get_config()

    # Init vars
    investor_type = config.investor_type
    initial_investment = config.initial_investment
    context_window_date = config.context_window_date
    investment_schedule = config.investment_schedule
    num_simulated_months = config.num_simulated_months
    num_simulations = config.num_simulations
    llm_additional_context = config.llm_additional_context
    experiment_folder_path = config.experiment_folder_path
    additional_context_dataset_path = config.additional_context_dataset_path
    additional_context_sample_size = config.additional_context_sample_size
    transaction_cost = config.transaction_cost

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
                print('New allocation', updated_portfolio)

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
                print("Current date", context_window_date)
                print("Current total value", portfolio_position['total_value'])
                print("Current portfolio value", portfolio_position['total_portfolio_value'])
                print("Current cash value", portfolio_position['cash_balance'])

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
    main(args.config)