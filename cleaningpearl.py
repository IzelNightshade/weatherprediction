import pandas as pd

df = pd.read_csv('PEARL.csv')



df.set_index('Date', inplace=True)
df = df.iloc[:, :-2]
df.drop('Remarks', axis=1, inplace=True)
df.drop('Pass Time (Local)', axis=1, inplace=True)
df.drop('Satellite', axis=1, inplace=True)
df.index = pd.to_datetime(df.index, format='%d %B %Y')

df["target"] = df.shift(-1)["Temperature"]
df = df.ffill()

print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming 'df' is your dfFrame with 'Temperature' and 'target' columns
# Convert the index to datetime if it's not already in datetime format
df.index = pd.to_datetime(df.index)

# Split the df into features (Temperature) and target (next day's temperature)
X = df[['Temperature']]
y = df['target']

# Split the df into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model (for example, using Mean Squared Error)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
