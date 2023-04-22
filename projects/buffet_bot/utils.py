import openai
from dateutil.relativedelta import relativedelta
import datetime
import yfinance as yf
import json

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

   response = bot.get_response(llm_prompt)
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
   context_window_date_obj = datetime.datetime.strptime(context_window_date, '%Y-%m-%d')
   end_date_obj = context_window_date_obj + relativedelta(months=1)
   end_date = end_date_obj.strftime('%Y-%m-%d')

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
      context_window_date_obj = datetime.datetime.strptime(context_window_date, '%Y-%m-%d')
      context_window_date_obj = context_window_date_obj + relativedelta(months=1)
      context_window_date = context_window_date_obj.strftime('%Y-%m-%d')
   return context_window_date