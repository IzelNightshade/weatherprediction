import pandas as pd


# Replace 'your_file.csv' with the actual path to your CSV file


# Read the CSV file into a DataFrame
weather = pd.read_csv('input.csv')





# Replace missing values in a specific column (replace 'column_name' with the actual column name)
column_name = 'PRCP'
avg_value = weather[column_name].mean()

# Use fillna() to replace missing values with the average
weather[column_name].fillna(avg_value, inplace=True)


# check null percentage
# null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]*100
# print(null_pct)

# Use the drop() method to delete the specified column
weather.drop('TMAX', axis=1, inplace=True)
weather.drop('TMIN', axis=1, inplace=True)

# Specify the column to be set as the new index (replace 'DATW' with the actual column name)
new_index_column = 'DATE'

# Use set_index() to set the specified column as the new index
weather.set_index(new_index_column, inplace=True)

weather.index = pd.to_datetime(weather.index)
weather.index = weather.index.strftime('%d-%m-%Y')


# Display the DataFrame after deleting the column
print(weather)


# print(weather.dtypes)


weather.to_csv('cleaned.csv', index=True)