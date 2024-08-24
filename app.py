from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# Initialize the Flask application
app = Flask(__name__)

# Load and preprocess the weather data
weather = pd.read_csv('cleaned.csv')
weather['DATE'] = pd.to_datetime(weather['DATE'], format='%d-%m-%Y')
weather[['TAVG', 'TMAX', 'TMIN']] = weather[['TAVG', 'TMAX', 'TMIN']] / 10
weather.set_index('DATE', inplace=True)

# Create new features
weather['day_of_year'] = weather.index.dayofyear
weather['month'] = weather.index.month
weather['week'] = weather.index.isocalendar().week
weather['lag1_TAVG'] = weather['TAVG'].shift(1)
weather['lag7_TAVG'] = weather['TAVG'].shift(7)
weather['rolling_mean7_TAVG'] = weather['TAVG'].rolling(window=7).mean()
weather = weather.ffill()

# Define predictors and targets
predictors = ['day_of_year', 'month', 'week', 'lag1_TAVG', 'lag7_TAVG', 'rolling_mean7_TAVG']
target_columns = ['TAVG', 'TMAX', 'TMIN']

# Train models for each target
models = {}
for target in target_columns:
    target_col = f'target_{target}'
    weather[target_col] = weather[target].shift(-1)
    weather = weather.ffill()
    train = weather.dropna()
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(train[predictors], train[target_col])
    models[target] = model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])


def predict():
    date_input = request.form['date']
    date = datetime.strptime(date_input, '%Y-%m-%d')
    
    dates = [date + timedelta(days=i) for i in range(7)]
    new_data = pd.DataFrame(index=dates, columns=predictors)
    
    for col in predictors:
        new_data[col] = weather[col].iloc[-1]  # Use the last available data to fill the new data
    
    new_data = new_data.ffill().bfill()
    
    predictions = {}
    for target in target_columns:
        model = models[target]
        pred = model.predict(new_data[predictors])
        predictions[target] = pred
    
    prediction_results = pd.DataFrame({
        'DATE': dates,
        'Predicted TAVG': predictions['TAVG'],
        'Predicted TMAX': predictions['TMAX'],
        'Predicted TMIN': predictions['TMIN']
    })

    # Pass the DataFrame directly to the template
    return render_template('results.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
