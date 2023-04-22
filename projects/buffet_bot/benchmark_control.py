import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Set the investment details
initial_investment = 100000
start_date = "2018-01-01"
end_date = "2022-01-01"

# Download S&P 500 data from Yahoo Finance
ticker = "^GSPC"
sp500 = yf.download(ticker, start=start_date, end=end_date)

# Calculate investment performance
sp500["Normalized"] = sp500["Adj Close"] / sp500["Adj Close"].iloc[0]
sp500["Investment"] = sp500["Normalized"] * initial_investment

# Plot the performance
plt.figure(figsize=(10, 6))
plt.plot(sp500.index, sp500["Investment"])
plt.xlabel("Date")
plt.ylabel("Investment Value ($)")
plt.title(
    f"Performance of ${initial_investment} Investment in S&P 500\n({start_date} to {end_date})"
)
plt.grid()
# plt.show()

# Save the graph to png
plt.savefig("output/experiments/control/result.png")
