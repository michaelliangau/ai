import datetime
import glob
import json
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import IPython


def get_average_values(folder):
    """Get the average values for each date from a folder of experiments.

    Args:
        folder (str): The folder name.

    Returns:
        average_dates (list): A list of datetime objects.
        average_values_list (list): A list of average values.
    """
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


def calculate_sharpe_ratio(returns, risk_free_rate):
    """Calculate the Sharpe ratio of a list of returns.

    Sharpe ratio is scaled by the square root of the sample size.

    Args:
        returns (list): A list of returns.
        risk_free_rate (float): The risk-free rate.
    """
    excess_returns = returns - risk_free_rate
    n = len(excess_returns)
    # Ensure there are enough data points for calculation
    if n > 0:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(n)
    else:
        sharpe_ratio = 0
    return sharpe_ratio


folders = [
    "value_no_news_context",
    "growth_no_news_context",
    "news_context_ss_200_filtered_sharpe",
    "news_context_ss_200_filtered_growth",
    "news_context_ss_200_filtered_value_x_large",
    "news_context_ss_200_filtered_value_large",
    "news_context_ss_200_filtered_value",
    "news_context_ss_200",
    "news_context",
    "no_temp_no_context_4_year",
]  # Add your folder names here
folder_labels = [
    "Claude Value No News Context",
    "Claude Growth No News Context",
    "Claude With News Context SS200 Filtered DS High Sharpe",
    "Claude With News Context SS200 Filtered DS Growth Investor",
    "Claude With News Context SS200 Filtered DS Value Extra Large Investor",
    "Claude With News Context SS200 Filtered DS Value Large Investor",
    "Claude With News Context SS200 Filtered DS Value Investor",
    "Claude With News Context SS200",
    "Claude With News Context SS100",
    "Claude Base",
]  # Add your desired legend names for folders here

# Vars
start_date = "2018-01-01"
end_date = "2022-01-01"
initial_investment = 100000

# Calculate S&P 500 investment performance
ticker = "^GSPC"
sp500 = yf.download(ticker, start=start_date, end=end_date)

# Download Nasdaq 100 data from Yahoo Finance
nasdaq_ticker = "^NDX"
nasdaq100 = yf.download(nasdaq_ticker, start=start_date, end=end_date)

# Download DJIA data from Yahoo Finance
djia_ticker = "^DJI"
djia = yf.download(djia_ticker, start=start_date, end=end_date)

# Download FTSE 100 data from Yahoo Finance
ftse_ticker = "^FTSE"
ftse100 = yf.download(ftse_ticker, start=start_date, end=end_date)

# Resample index data to monthly frequency
sp500_monthly = sp500.resample("M").ffill()
nasdaq100_monthly = nasdaq100.resample("M").ffill()
djia_monthly = djia.resample("M").ffill()
ftse100_monthly = ftse100.resample("M").ffill()

# Calculate investment performance
sp500_monthly["Normalized"] = (
    sp500_monthly["Adj Close"] / sp500_monthly["Adj Close"].iloc[0]
)
sp500_monthly["Investment"] = sp500_monthly["Normalized"] * initial_investment

nasdaq100_monthly["Normalized"] = (
    nasdaq100_monthly["Adj Close"] / nasdaq100_monthly["Adj Close"].iloc[0]
)
nasdaq100_monthly["Investment"] = nasdaq100_monthly["Normalized"] * initial_investment

djia_monthly["Normalized"] = (
    djia_monthly["Adj Close"] / djia_monthly["Adj Close"].iloc[0]
)
djia_monthly["Investment"] = djia_monthly["Normalized"] * initial_investment

ftse100_monthly["Normalized"] = (
    ftse100_monthly["Adj Close"] / ftse100_monthly["Adj Close"].iloc[0]
)
ftse100_monthly["Investment"] = ftse100_monthly["Normalized"] * initial_investment

# Calculate returns
sp500_returns = (
    np.diff(sp500_monthly["Investment"].dropna())
    / sp500_monthly["Investment"].dropna()[:-1]
)
nasdaq100_returns = (
    np.diff(nasdaq100_monthly["Investment"].dropna())
    / nasdaq100_monthly["Investment"].dropna()[:-1]
)
djia_returns = (
    np.diff(djia_monthly["Investment"].dropna())
    / djia_monthly["Investment"].dropna()[:-1]
)
ftse100_returns = (
    np.diff(ftse100_monthly["Investment"].dropna())
    / ftse100_monthly["Investment"].dropna()[:-1]
)

# Calculate Sharpe ratios for the indices
index_returns = {
    "S&P 500": sp500_returns,
    "Nasdaq 100": nasdaq100_returns,
    "DJIA": djia_returns,
    "FTSE 100": ftse100_returns,
}

# Create a figure and axis for the line graph
fig, ax = plt.subplots()

# Configure the date format
date_fmt = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(date_fmt)

# Plot the average line graphs for each folder
for folder, folder_label in zip(folders, folder_labels):
    average_dates, average_values_list = get_average_values(folder)
    ax.plot(
        average_dates,
        average_values_list,
        linestyle="--",
        linewidth=2,
        label=folder_label,
    )

# Plot the S&P 500 investment performance
ax.plot(
    sp500_monthly.index,
    sp500_monthly["Investment"],
    linestyle="-",
    linewidth=2,
    label="S&P 500",
)

# Plot the Nasdaq 100 investment performance
ax.plot(
    nasdaq100_monthly.index,
    nasdaq100_monthly["Investment"],
    linestyle="-.",
    linewidth=2,
    label="Nasdaq 100",
)

# Plot the DJIA investment performance
ax.plot(
    djia_monthly.index,
    djia_monthly["Investment"],
    linestyle=":",
    linewidth=2,
    label="DJIA",
)

# Plot the FTSE 100 investment performance
ax.plot(
    ftse100_monthly.index,
    ftse100_monthly["Investment"],
    linestyle="--",
    linewidth=1,
    label="FTSE 100",
)

# Sharpe Ratio
# Calculate the average risk-free rate for the period
risk_free_rate_annual = 0.03  # 3%
risk_free_rate_monthly = (
    risk_free_rate_annual / 12
)  # Convert the annual rate to a monthly rate

sharpe_ratios = []

for folder, folder_label in zip(folders, folder_labels):
    average_dates, average_values_list = get_average_values(folder)
    returns = np.diff(average_values_list) / average_values_list[:-1]
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate_monthly)
    sharpe_ratios.append((folder_label, sharpe_ratio))

index_sharpe_ratios = []
for index_name, index_return in index_returns.items():
    sharpe_ratio = calculate_sharpe_ratio(index_return, risk_free_rate_monthly)
    index_sharpe_ratios.append((index_name, sharpe_ratio))

# Create the Sharpe ratios text
sharpe_ratios_text = [
    f"{label} (Sharpe: {sharpe_ratio:.2f})"
    for label, sharpe_ratio in sharpe_ratios + index_sharpe_ratios
]

# Customize graph
plt.xlabel("Dates")
plt.ylabel("Total Values")
plt.title("Comparison of Average Performances and Major Stock Indices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(sharpe_ratios_text)

# Show the graph
plt.show()

# Save the graph
fig.savefig(f"output/all_experiments_result_with_major_indices.png")
