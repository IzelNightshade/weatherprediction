import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

# Load the weather data
weather = pd.read_csv('cleaned.csv')

# Specify the column to be set as the new index
new_index_column = 'DATE'

# Convert the 'DATE' column to datetime format
weather[new_index_column] = pd.to_datetime(weather[new_index_column], format='%d-%m-%Y')
weather[['TAVG', 'TMAX', 'TMIN']] = weather[['TAVG', 'TMAX', 'TMIN']] / 10

# Create new features
weather['day_of_year'] = weather[new_index_column].dt.dayofyear
weather['month'] = weather[new_index_column].dt.month
weather['week'] = weather[new_index_column].dt.isocalendar().week
weather['lag1_TAVG'] = weather['TAVG'].shift(1)
weather['lag7_TAVG'] = weather['TAVG'].shift(7)
weather['rolling_mean7_TAVG'] = weather['TAVG'].rolling(window=7).mean()
weather = weather.ffill()

# Set the 'DATE' column as the new index
weather.set_index(new_index_column, inplace=True)

# Create the target columns
weather["target_TAVG"] = weather["TAVG"].shift(-1)
weather["target_TMAX"] = weather["TMAX"].shift(-1)
weather["target_TMIN"] = weather["TMIN"].shift(-1)
weather = weather.ffill()

# Define the predictors
predictors = weather.columns[~weather.columns.isin(["target_TAVG", "target_TMAX", "target_TMIN", "NAME", "STATION"])]

# Split data into training and testing sets
train, test = train_test_split(weather.dropna(), test_size=0.2, shuffle=False)

# Train Gradient Boosting models for each target
models = {}
for target in ["target_TAVG", "target_TMAX", "target_TMIN"]:
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(train[predictors], train[target])
    models[target] = model

# Predict on the test set for each target
predictions = {}
for target in ["target_TAVG", "target_TMAX", "target_TMIN"]:
    pred = models[target].predict(test[predictors])
    predictions[target] = pred

# Evaluate the models
for target in ["target_TAVG", "target_TMAX", "target_TMIN"]:
    mae = mean_absolute_error(test[target], predictions[target])
    mse = mean_squared_error(test[target], predictions[target])
    r2 = r2_score(test[target], predictions[target])
    print(f"{target} - MAE: {mae}, MSE: {mse}, R2: {r2}")

# Function to get user input date and make predictions
def get_predictions_for_date(models, weather, predictors):
    date_input = input("Enter a date (dd/mm/yyyy): ")
    date = datetime.strptime(date_input, '%d/%m/%Y')
    
    dates = [date + timedelta(days=i) for i in range(7)]
    new_data = pd.DataFrame(index=dates, columns=predictors)
    
    for col in predictors:
        if col in weather.columns:
            new_data[col] = weather.loc[:date, col].iloc[-1]
    
    new_data = new_data.ffill().bfill()
    
    predicted_tavg = models["target_TAVG"].predict(new_data[predictors])
    predicted_tmax = models["target_TMAX"].predict(new_data[predictors])
    predicted_tmin = models["target_TMIN"].predict(new_data[predictors])
    
    prediction_results = pd.DataFrame({
        'DATE': dates,
        'Predicted TAVG': predicted_tavg,
        'Predicted TMAX': predicted_tmax,
        'Predicted TMIN': predicted_tmin
    })
    
    return prediction_results

# Prompt user for date and display predictions using the Gradient Boosting model
predicted_temperatures = get_predictions_for_date(models, weather, predictors)
print(predicted_temperatures)
