import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import IPython
import numpy as np

# Read results from output folder
results_list = []
for filename in glob.glob('output/no_temp_no_context/*.json'):
    with open(filename, 'r') as f:
        result = json.load(f)
        results_list.append(result)

# Create a figure and axis for the line graph
fig, ax = plt.subplots()

# Configure the date format
date_fmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_fmt)

# Prepare data structure for the average values
average_values = {}

# Loop through the results_list and plot the line graph for each item
for result in results_list:
    dates = []
    values = []
    for data_point in result:
        date = datetime.datetime.strptime(data_point['date'], '%Y-%m-%d')
        value = data_point['total_value']
        dates.append(date)
        values.append(value)

        # Add values to the average_values dictionary
        if date not in average_values:
            average_values[date] = []
        average_values[date].append(value)

    ax.plot(dates, values)

# Calculate the average for each date and plot the average line
average_dates = sorted(average_values.keys())
average_values_list = [np.mean(average_values[date]) for date in average_dates]
ax.plot(average_dates, average_values_list, linestyle='--', linewidth=2, label='Average')

# Customize graph
plt.xlabel('Dates')
plt.ylabel('Total Values')
plt.title('No Temperature with No Context')
plt.legend([f'Result {i+1}' for i in range(len(results_list))] + ['Average'])
plt.xticks(rotation=45)
plt.tight_layout()

# Show the graph
# plt.show()

# Save the graph
fig.savefig('output/no_temp_no_context/result.png')