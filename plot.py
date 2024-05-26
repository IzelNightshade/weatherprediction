import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('predictions.csv')

# Assuming 'DATE' is in datetime format already, else convert it
data['DATE'] = pd.to_datetime(data['DATE'])

# Plotting the 'actual' and 'prediction' lines for a specific date range
start_date = '2022-01-01'  # Change this to your desired start date
end_date = '2022-01-31'    # Change this to your desired end date

# Filter the data for the specified date range
filtered_data = data[(data['DATE'] >= start_date) & (data['DATE'] <= end_date)]

plt.figure(figsize=(10, 6))
plt.plot(filtered_data['DATE'], filtered_data['actual'], label='Actual')
plt.plot(filtered_data['DATE'], filtered_data['prediction'], label='Prediction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Prediction')
plt.legend()
plt.show()




