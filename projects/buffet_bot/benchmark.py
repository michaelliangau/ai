import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import yfinance as yf
import numpy as np

# Vars
folder = 'no_temp_no_context_4_year'

# Read results from output folder
results_list = []
for filename in glob.glob(f'output/experiments/{folder}/*.json'):
    with open(filename, 'r') as f:
        result = json.load(f)
        results_list.append(result)

# Download S&P 500 data from Yahoo Finance
start_date = '2018-01-01'
end_date = '2022-01-01'
ticker = '^GSPC'
sp500 = yf.download(ticker, start=start_date, end=end_date)

# Calculate S&P 500 investment performance
initial_investment = 100000
sp500['Normalized'] = sp500['Adj Close'] / sp500['Adj Close'].iloc[0]
sp500['Investment'] = sp500['Normalized'] * initial_investment

# Create a figure and axis for the line graph
fig, ax = plt.subplots()

# Configure the date format
date_fmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_fmt)

# Prepare data structure for the average values
average_values = {}

# Loop through the results_list and calculate average values
for result in results_list:
    for data_point in result:
        date = datetime.datetime.strptime(data_point['date'], '%Y-%m-%d')
        value = data_point['total_value']

        # Add values to the average_values dictionary
        if date not in average_values:
            average_values[date] = []
        average_values[date].append(value)

# Calculate the average for each date and plot the average line
average_dates = sorted(average_values.keys())
average_values_list = [np.mean(average_values[date]) for date in average_dates]
ax.plot(average_dates, average_values_list, linestyle='--', linewidth=2, label='Average')

# Plot the S&P 500 investment performance
ax.plot(sp500.index, sp500['Investment'], linestyle='-', linewidth=2, label='S&P 500')

# Customize graph
plt.xlabel('Dates')
plt.ylabel('Total Values')
plt.title('No Temperature with No Context')
plt.legend(['Average', 'S&P 500'])
plt.xticks(rotation=45)
plt.tight_layout()

# Show the graph
plt.show()

# Save the graph
fig.savefig(f'output/experiments/{folder}/result_with_sp500.png')
