import pandas as pd

# Read the CSV file into a DataFrame
weather = pd.read_csv('input.csv')

# Replace missing values in a specific column (replace 'column_name' with the actual column name)
column_name = 'PRCP'
avg_value = weather[column_name].mean()

# Use fillna() to replace missing values with the average
weather[column_name].fillna(avg_value, inplace=True)

# Check null percentage
# null_pct = weather.apply(pd.isnull).sum() / weather.shape[0] * 100
# print(null_pct)

# Drop unnecessary columns
weather.drop(['PRCP_ATTRIBUTES', 'TAVG_ATTRIBUTES', 'TMAX_ATTRIBUTES', 'TMIN_ATTRIBUTES'], axis=1, inplace=True)

# Specify the column to be set as the new index
new_index_column = 'DATE'

# Use set_index() to set the specified column as the new index
weather.set_index(new_index_column, inplace=True)

weather.index = pd.to_datetime(weather.index)
weather.index = weather.index.strftime('%d-%m-%Y')

# Display the DataFrame after deleting the column
print(weather)

# Save the cleaned DataFrame to a CSV file
weather.to_csv('cleaned.csv', index=True)
