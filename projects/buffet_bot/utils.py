import re
import openai
from dateutil.relativedelta import relativedelta
import datetime
import json
from datetime import datetime
import random
import IPython
import yfinance as yf

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def find_next_trading_day(context_window_date, simulator, ticker="MSFT"):
    """Finds the next trading day for the given ticker after the given date.

    Args:
       context_window_date (str): The date to find the next trading day.
       simulator (StockSimulator): The stock simulator.
       ticker (str, optional): The ticker to use. Defaults to 'MSFT'.

    Returns:
       context_window_date (str): The next trading day."""
    while not simulator.is_trading_day(ticker, date=context_window_date):
        context_window_date = simulator.find_next_trading_day(
            ticker, start_date=context_window_date
        )
    return context_window_date

valid_tickers_cache = set()
invalid_tickers_cache = {'DOW'}

def check_tickers_exist(tickers):
    """Checks if the given tickers exist.

    Args:
        tickers (list): The list of tickers to check.

    Returns:
        valid_tickers (list): A list of tickers that exist.
    """
    global valid_tickers_cache, invalid_tickers_cache

    # Filter tickers that are not in the valid or invalid cache
    tickers_to_check = [ticker for ticker in tickers if ticker not in valid_tickers_cache and ticker not in invalid_tickers_cache]

    if tickers_to_check:
        tickers_string = ' '.join(tickers_to_check)
        stock_data = yf.download(tickers_string, period='1d', start='2018-01-01', end='2022-01-01')

        for ticker in tickers_to_check:
            if stock_data['Adj Close'][ticker].isna().all():
                invalid_tickers_cache.add(ticker)
            else:
                valid_tickers_cache.add(ticker)

    valid_tickers = [ticker for ticker in tickers if ticker in valid_tickers_cache]

    return valid_tickers

def clean_portfolio(portfolio):
    """Cleans the portfolio by removing tickers that don't exist.

    Args:
        portfolio (dict): The portfolio with ticker symbols as keys and allocation percentages as values.

    Returns:
        cleaned_portfolio (dict): The cleaned portfolio.
    """
    # Replace periods with hyphens in all tickers
    portfolio = {ticker.replace('.', '-'): percentage for ticker, percentage in portfolio.items()}
    
    tickers = list(portfolio.keys())
    valid_tickers = check_tickers_exist(tickers)

    cleaned_portfolio = {}
    for ticker, percentage in portfolio.items():
        if ticker in valid_tickers:
            cleaned_portfolio[ticker] = percentage

    return rebalance_portfolio(cleaned_portfolio)



def rebalance_portfolio(portfolio, original_total_allocation=None):
    """Rebalances the portfolio to ensure the sum of all values is equal to the original total allocation or 100.

    Args:
        portfolio (dict): The portfolio with ticker symbols as keys and allocation percentages as values.
        original_total_allocation (float): The original total allocation before cleaning the portfolio.

    Returns:
        rebalanced_portfolio (dict): The rebalanced portfolio with the sum of all values equal to the original total allocation or 100.
    """
    # Calculate the total allocation
    total_allocation = sum(portfolio.values())

    if original_total_allocation is None:
        original_total_allocation = total_allocation

    # Scale the allocation values to add up to the original total allocation
    rebalanced_portfolio = {
        ticker: allocation * (original_total_allocation / total_allocation)
        for ticker, allocation in portfolio.items()
    }

    # Check if the new total allocation is above 100
    new_total_allocation = sum(rebalanced_portfolio.values())
    if new_total_allocation > 100:
        # Scale the allocation values to add up to 100
        rebalanced_portfolio = {
            ticker: allocation * (100 / new_total_allocation)
            for ticker, allocation in rebalanced_portfolio.items()
        }

    return rebalanced_portfolio


def get_llm_response(bot, investor_type, context_window_date, current_holdings):
    """Gets the response from the LLM and returns the updated portfolio.

    Args:
       bot (openai.api.DefaultApi): The bot to use.
       investor_type (str): The type of investor changes the llm prompt.
       context_window_date (str): The date to use as the context window.
       current_holdings (dict): The current holdings.

    Returns:
       updated_portfolio (dict): The updated portfolio.
    """
    try:
        if investor_type == "value":
            llm_prompt = f"You are Warren Buffett, one of the world's most successful value investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Warren Buffett, you have a portfolio to invest in any investment vehicle. Build your portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding (integer percentages) in JSON format. Your percentage holdings should not exceed 100%. Just return the JSON object, this is very important."
        elif investor_type == "growth":
            llm_prompt = f"You are Peter Lynch, a proficient growth investor with a deep understanding of the stock market and a long history of making well-informed investment decisions. Your goal is to build a portfolio that focuses on companies with strong financials, high-quality management, competitive advantages, and a reasonable margin of safety. As Peter Lynch, you prioritize long-term capital appreciation over short-term gains. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding (integer percentages) in JSON format. Your percentage holdings should not exceed 100%. Just return the JSON object, this is very important."
        elif investor_type == "value_large":
            llm_prompt = f"You are Warren Buffett, one of the world's most successful value investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Warren Buffett, you have a portfolio to invest in any investment vehicle. Build your portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding (integer percentages) in JSON format, aim to diversify between 10 to 30 companies. Your percentage holdings should not exceed 100%. Just return the JSON object, this is very important."
        elif investor_type == "value_x_large":
            llm_prompt = f"You are Warren Buffett, one of the world's most successful value investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Warren Buffett, you have a portfolio to invest in any investment vehicle. Build your portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding (integer percentages) in JSON format, aim to diversify between 30 to 100 companies. Your percentage holdings should not exceed 100%. Just return the JSON object, this is very important."
        elif investor_type == "sharpe":
            llm_prompt = f"You are Ray Dalio, one of the world's most successful investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Ray Dalio, you have a portfolio with high Sharpe Ratio to invest in any investment vehicle. Build your high Sharpe Ratio portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding (integer percentages) in JSON format. Your percentage holdings should not exceed 100%. Just return the JSON object, this is very important."

        response = bot.get_response(llm_prompt, context_window_date)
        response["completion"] = response["completion"].replace("'", '"')
        # Change FB to META
        response["completion"] = response["completion"].replace("FB", "META")
        # Fix output in case it doesn't just return JSON.
        response["completion"] = re.search(
            r"\{[^}]*\}", response["completion"], re.DOTALL
        ).group()
        updated_portfolio = json.loads(response["completion"])
        updated_portfolio = clean_portfolio(updated_portfolio)

        return updated_portfolio

    except Exception as e:
        print(e)
        IPython.embed()


def update_holdings(
    simulator,
    updated_portfolio,
    context_window_date,
    initial_investment,
    prev_updated_portfolio,
):
    """Updates the holdings in the simulator based on the updated portfolio.

    Args:
       simulator (StockSimulator): The stock simulator.
       updated_portfolio (dict): The updated portfolio.
       context_window_date (str): The date to use as the context window.
       initial_investment (int): The initial investment.
       prev_updated_portfolio (dict): The previous updated portfolio.

    Returns:
       prev_updated_portfolio (dict): The previous updated portfolio.
    """
    end_date = add_one_month(context_window_date)

    for ticker in updated_portfolio.keys():
        simulator.get_stock_data(
            ticker, start_date=context_window_date, end_date=end_date
        )

    if updated_portfolio != prev_updated_portfolio:
        simulator.update_holdings(
            updated_portfolio, context_window_date, initial_investment
        )

    # Update prev_updated_portfolio
    prev_updated_portfolio = updated_portfolio

    return prev_updated_portfolio


def increment_time(investment_schedule, context_window_date):
    """Increments the context window date based on the investment schedule to the first of the next month.

    Args:
       investment_schedule (str): The investment schedule.
       context_window_date (str): The date to use as the context window.

    Returns:
       context_window_date (str): The updated context window date.
    """
    if investment_schedule == "monthly":
        # Increment context_start_date by 1 month
        context_window_date = add_one_month(context_window_date)
        context_window_date = set_day_to_first(context_window_date)

    return context_window_date


def get_headlines_between_dates(
    file_path,
    start_date,
    end_date,
    additional_context_sample_size,
    use_impact_score=True,
):
    """Gets the headlines between the given dates.

    Args:
       file_path (str): The file path to the headlines.
       start_date (str): The start date.
       end_date (str): The end date.
       additional_context_sample_size (int, optional): The sample size for headlines if there is too many.
       use_impact_score (bool, optional): Use the highest impact scores for sampling if True.

    Returns:
       headlines (str): The headlines between the given dates in a newline-separated format.
    """
    headlines_list = []

    # Convert input dates to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    with open(file_path, "r") as file:
        for line in file:
            # Load JSON object from the line
            news_item = json.loads(line)

            # Convert the 'date' field to a datetime object
            date_obj = datetime.strptime(news_item["date"], "%Y-%m-%d")

            # Check if the date is within the specified range
            if start_date_obj <= date_obj < end_date_obj:
                headlines_list.append(news_item)

    # Sample headlines based on impact score or randomly
    if len(headlines_list) > additional_context_sample_size:
        if use_impact_score:
            headlines_list = sorted(
                headlines_list, key=lambda x: x["impact_score"], reverse=True
            )
            headlines_list = headlines_list[:additional_context_sample_size]
        else:
            headlines_list = random.sample(
                headlines_list, additional_context_sample_size
            )

    # Convert the list of headlines into a newline-separated string
    headlines = "\n".join([item["headline"] for item in headlines_list])

    return headlines


def set_day_to_first(date_str):
    """Sets the day to the first day of the month.

    Args:
       date_str (str): The date to set the day to the first.

    Returns:
       first_day (str): The date with the day set to the first.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    first_day_obj = date_obj.replace(day=1)
    first_day = first_day_obj.strftime("%Y-%m-%d")
    return first_day


def add_one_month(date_str):
    """Adds one month to the given date.

    Args:
       date_str (str): The date to add one month to.

    Returns:
       end_date (str): The date after adding one month.
    """
    start_date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    end_date_obj = start_date_obj + relativedelta(months=1)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    return end_date


def subtract_one_month(date_str):
    """Subtracts one month from the given date.

    Args:
       date_str (str): The date to subtract one month from.

    Returns:
       end_date (str): The date after subtracting one month.
    """
    start_date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    end_date_obj = start_date_obj - relativedelta(months=1)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    return end_date
