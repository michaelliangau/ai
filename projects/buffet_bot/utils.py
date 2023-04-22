import openai
from dateutil.relativedelta import relativedelta
import datetime
import yfinance as yf
import json
from datetime import datetime
import random

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def find_next_trading_day(context_window_date, simulator, ticker='MSFT'):
   """Finds the next trading day for the given ticker after the given date.
   
   Args:
      context_window_date (str): The date to find the next trading day.
      simulator (StockSimulator): The stock simulator.
      ticker (str, optional): The ticker to use. Defaults to 'MSFT'.
      
   Returns:
      context_window_date (str): The next trading day."""
   while not simulator.is_trading_day(ticker, date=context_window_date):
      context_window_date = simulator.find_next_trading_day(ticker, start_date=context_window_date)
   return context_window_date

def get_llm_response(bot, investor_type, context_window_date, current_holdings):
   """Gets the response from the LLM and returns the updated portfolio.

   Args:
      bot (openai.api.DefaultApi): The bot to use.
      investor_type (str): The type of investor.
      context_window_date (str): The date to use as the context window.
      current_holdings (dict): The current holdings.
   
   Returns:
      updated_portfolio (dict): The updated portfolio.
   """
   if investor_type == 'value':
      llm_prompt = f'You are Warren Buffett, one of the world\'s most successful value investors, with a deep understanding of the stock market and a long history of making well-informed investment decisions. As Warren Buffett, you have a portfolio to invest in any investment vehicle. Build your portfolio and in your recommendations, consider factors such as company financials, management quality, competitive advantages, and the margin of safety. You do not have knowledge of events after {context_window_date}. Your current portfolio holdings in JSON format are: {current_holdings}. Always return only the ticker symbols of your investments and the percentage holding (integers percentages) in JSON format with double quotes. Do not return any other text.'

   response = bot.get_response(llm_prompt, context_window_date)
   response['completion'] = response['completion'].replace("'", '"')
   updated_portfolio = json.loads(response['completion'])

   return updated_portfolio

def update_holdings(simulator, updated_portfolio, context_window_date, initial_investment, prev_updated_portfolio):
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
   updated_portfolio = {key.replace(".", "-"): value for key, value in updated_portfolio.items()}
   end_date = add_one_month(context_window_date)

   for ticker in updated_portfolio.keys():
      simulator.get_stock_data(ticker, start_date=context_window_date, end_date=end_date)

   if updated_portfolio != prev_updated_portfolio:
      simulator.update_holdings(updated_portfolio, context_window_date, initial_investment)

   # Update prev_updated_portfolio
   prev_updated_portfolio = updated_portfolio

   return prev_updated_portfolio

def increment_time(investment_schedule, context_window_date):
   """Increments the context window date based on the investment schedule.
   
   Args:
      investment_schedule (str): The investment schedule.
      context_window_date (str): The date to use as the context window.
   
   Returns:
      context_window_date (str): The updated context window date.
   """
   if investment_schedule == 'monthly':
      # Increment context_start_date by 1 month
      context_window_date = add_one_month(context_window_date)
   return context_window_date

def get_headlines_between_dates(file_path, start_date, end_date, additional_context_sample_size):
   """Gets the headlines between the given dates.
   
   Args:
      file_path (str): The file path to the headlines.
      start_date (str): The start date.
      end_date (str): The end date.
      additional_context_sample_size (int, optional): The sample size for headlines if there is too many.
   
   Returns:
      headlines (str): The headlines between the given dates in a newline-separated format.
   """
   headlines_list = []

   # Convert input dates to datetime objects
   start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
   end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

   with open(file_path, 'r') as file:
      for line in file:
         # Load JSON object from the line
         news_item = json.loads(line)

         # Convert the 'date' field to a datetime object
         date_obj = datetime.strptime(news_item['date'], "%Y-%m-%d")

         # Check if the date is within the specified range
         if start_date_obj <= date_obj < end_date_obj:
               headlines_list.append(news_item['headline'])

   # Randomly sample headlines
   if len(headlines_list) > additional_context_sample_size:
      headlines_list = random.sample(headlines_list, additional_context_sample_size)

   # Convert the list of headlines into a newline-separated string
   headlines = "\n".join(headlines_list)

   return headlines

def add_one_month(date_str):
   """Adds one month to the given date.
   
   Args:
      date_str (str): The date to add one month to.
   
   Returns:
      end_date (str): The date after adding one month.
   """
   start_date_obj = datetime.strptime(date_str, "%Y-%m-%d")
   end_date_obj = start_date_obj + relativedelta(months=1)
   end_date = end_date_obj.strftime('%Y-%m-%d')
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
   end_date = end_date_obj.strftime('%Y-%m-%d')
   return end_date