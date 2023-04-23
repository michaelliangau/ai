# Native imports
import json
import argparse
import datetime

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
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="config/growth_config.py",
    help="Path to the configuration file.",
)
args = parser.parse_args()


def main(config_path: str):
    # Init configs
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.get_config()

    # Init vars
    real_trading = config.real_trading
    investor_type = config.investor_type
    initial_investment = config.initial_investment
    context_window_date = config.context_window_date
    investment_schedule = config.investment_schedule
    num_simulated_months = config.num_simulated_months
    num_simulations = config.num_simulations
    llm_additional_context = config.llm_additional_context
    transaction_cost = config.transaction_cost

    # Init bots and simulator
    if llm_additional_context == "news":
        bot = BuffetBot(
            llm="anthropic",
            additional_context=llm_additional_context,
            additional_context_sample_size=config.additional_context_sample_size,
            additional_context_dataset_path=config.additional_context_dataset_path,
        )
    else:
        bot = BuffetBot(llm="anthropic", additional_context=llm_additional_context)
    simulator = StockSimulator(initial_investment, real_trading)

    # Create output folder
    common_utils.create_folder(config.experiment_folder_path)
    
    if real_trading:
        today = utils.get_nyse_date()
        print("Today (NYSE) is", today)

        if not utils.is_nyse_trading_day(today):
            current_holdings = simulator.holdings

            # Get response from LLM
            updated_portfolio = utils.get_llm_response(
                bot, investor_type, today, current_holdings
            )
            print("New allocation", updated_portfolio)

            # Update holdings
            utils.update_holdings(
                simulator,
                updated_portfolio,
                today,
                initial_investment,
                None,
                transaction_cost,
            )

            # Print current portfolio position
            portfolio_position = simulator.get_portfolio_position(today)
            print("Current date", today)
            print("Current total value", portfolio_position["total_value"])
            print(
                "Current portfolio value", portfolio_position["total_portfolio_value"]
            )
            print("Current cash value", portfolio_position["cash_balance"])
        else:
            print("Not a trading day. No action taken.")

    else:
        # Run simulation
        for sim in range(num_simulations):
            results = []
            prev_updated_portfolio = None

            for _ in tqdm(range(num_simulated_months), total=num_simulated_months):
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
                    print("New allocation", updated_portfolio)

                    # If updated_portfolio is different from the previous one, update holdings
                    prev_updated_portfolio = utils.update_holdings(
                        simulator,
                        updated_portfolio,
                        context_window_date,
                        initial_investment,
                        prev_updated_portfolio,
                        transaction_cost,
                    )

                    # Print current portfolio position
                    portfolio_position = simulator.get_portfolio_position(
                        context_window_date
                    )
                    results.append(portfolio_position)
                    print("Current date", context_window_date)
                    print("Current total value", portfolio_position["total_value"])
                    print(
                        "Current portfolio value",
                        portfolio_position["total_portfolio_value"],
                    )
                    print("Current cash value", portfolio_position["cash_balance"])

                    # Increment time
                    context_window_date = utils.increment_time(
                        investment_schedule, context_window_date
                    )

                except Exception as e:
                    print(f"An error occurred: {e}")
                    traceback.print_exc()  # This line prints the full stack trace
                    IPython.embed()

            # Save the results
            with open(f"{config.experiment_folder_path}/sim_{sim}.json", "w") as f:
                json.dump(results, f)

            # Reset the context_window_date and simulator for the next simulation
            context_window_date = "2018-01-01"
            simulator.reset()


if __name__ == "__main__":
    main(args.config)
