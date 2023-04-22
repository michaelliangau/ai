import datetime
import glob
import json
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


def get_average_values(folder):
    results_list = []
    for filename in glob.glob(f"output/experiments/{folder}/*.json"):
        with open(filename, "r") as f:
            result = json.load(f)
            results_list.append(result)

    average_values = {}
    for result in results_list:
        for data_point in result:
            date = datetime.datetime.strptime(data_point["date"], "%Y-%m-%d")
            value = data_point["total_value"]

            if date not in average_values:
                average_values[date] = []
            average_values[date].append(value)

    average_dates = sorted(average_values.keys())
    average_values_list = [np.mean(average_values[date]) for date in average_dates]

    return average_dates, average_values_list


folders = [
    "news_context_ss_200_filtered_growth",
    "news_context_ss_200_filtered",
    "news_context_ss_200",
    "news_context",
    "no_temp_no_context_4_year",
]  # Add your folder names here
folder_labels = [
    "Claude Base With News Context SS200 Filtered DS Growth Investor",
    "Claude Base With News Context SS200 Filtered DS",
    "Claude Base With News Context SS200",
    "Claude Base With News Context SS100",
    "Claude Base",
]  # Add your desired legend names for folders here

# Download S&P 500 data from Yahoo Finance
start_date = "2018-01-01"
end_date = "2022-01-01"
ticker = "^GSPC"
sp500 = yf.download(ticker, start=start_date, end=end_date)

# Calculate S&P 500 investment performance
initial_investment = 100000
sp500["Normalized"] = sp500["Adj Close"] / sp500["Adj Close"].iloc[0]
sp500["Investment"] = sp500["Normalized"] * initial_investment

# Download Nasdaq 100 data from Yahoo Finance
nasdaq_ticker = "^NDX"
nasdaq100 = yf.download(nasdaq_ticker, start=start_date, end=end_date)

# Calculate Nasdaq 100 investment performance
nasdaq100["Normalized"] = nasdaq100["Adj Close"] / nasdaq100["Adj Close"].iloc[0]
nasdaq100["Investment"] = nasdaq100["Normalized"] * initial_investment

# Download DJIA data from Yahoo Finance
djia_ticker = '^DJI'
djia = yf.download(djia_ticker, start=start_date, end=end_date)

# Calculate DJIA investment performance
djia['Normalized'] = djia['Adj Close'] / djia['Adj Close'].iloc[0]
djia['Investment'] = djia['Normalized'] * initial_investment

# Download FTSE 100 data from Yahoo Finance
ftse_ticker = '^FTSE'
ftse100 = yf.download(ftse_ticker, start=start_date, end=end_date)

# Calculate FTSE 100 investment performance
ftse100['Normalized'] = ftse100['Adj Close'] / ftse100['Adj Close'].iloc[0]
ftse100['Investment'] = ftse100['Normalized'] * initial_investment

# Create a figure and axis for the line graph
fig, ax = plt.subplots()

# Configure the date format
date_fmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_fmt)

# Plot the average line graphs for each folder
for folder, folder_label in zip(folders, folder_labels):
    average_dates, average_values_list = get_average_values(folder)
    ax.plot(average_dates, average_values_list, linestyle='--', linewidth=2, label=folder_label)

# Plot the S&P 500 investment performance
ax.plot(sp500.index, sp500['Investment'], linestyle='-', linewidth=2, label='S&P 500')

# Plot the Nasdaq 100 investment performance
ax.plot(nasdaq100.index, nasdaq100['Investment'], linestyle='-.', linewidth=2, label='Nasdaq 100')

# Plot the DJIA investment performance
ax.plot(djia.index, djia['Investment'], linestyle=':', linewidth=2, label='DJIA')

# Plot the FTSE 100 investment performance
ax.plot(ftse100.index, ftse100['Investment'], linestyle='--', linewidth=1, label='FTSE 100')

# Customize graph
plt.xlabel('Dates')
plt.ylabel('Total Values')
plt.title('Comparison of Average Performances and Major Stock Indices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the graph
plt.show()

# Save the graph
fig.savefig(f'output/all_experiments_result_with_major_indices.png')
