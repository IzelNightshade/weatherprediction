from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

# Initialize the Flask application
app = Flask(__name__)

def create_date_features(date):
    return pd.Series({
        'day_of_year': date.dayofyear,
        'day_of_month': date.day,
        'month': date.month,
        'year': date.year,
        'is_weekend': date.weekday() >= 5,
    })

# Load and preprocess the weather data
weather = pd.read_csv('cleaned.csv')
weather['DATE'] = pd.to_datetime(weather['DATE'], format='%d-%m-%Y')
weather.set_index('DATE', inplace=True)
weather[['TAVG', 'TMAX', 'TMIN']] = weather[['TAVG', 'TMAX', 'TMIN']] / 10  # Convert to Celsius

# Create date features for training data
date_features = weather.index.to_series().apply(create_date_features)
weather = pd.concat([weather, date_features], axis=1)

# Prepare features and target
target = weather['TAVG']
features = weather[['PRCP', 'TMAX', 'TMIN', 'day_of_year', 'day_of_month', 'month', 'year', 'is_weekend']]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
imputed_features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns, index=features.index)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)

# Train the model
model = Ridge(alpha=0.1)
model.fit(scaled_features, target.dropna())

def predict_temperature_for_week(start_date):
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    predictions = []

    for i in range(7):
        current_date = start_date + timedelta(days=i)
        input_features = create_date_features(current_date).to_frame().T
        
        # Use historical averages for weather features
        historical_data = weather[weather.index.month == current_date.month]
        input_features['PRCP'] = historical_data['PRCP'].mean()
        input_features['TMAX'] = historical_data['TMAX'].mean()
        input_features['TMIN'] = historical_data['TMIN'].mean()
        
        # Ensure there are no missing values
        imputed_input = imputer.transform(input_features[features.columns])
        scaled_input = scaler.transform(imputed_input)
        prediction = model.predict(scaled_input)[0]
        
        # Handle undefined or non-numeric predictions
        if not isinstance(prediction, (int, float)):
            prediction = 0.0  # or some default value
        
        predictions.append((current_date, round(prediction, 2)))

    return predictions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date_input = request.form['date']
    weekly_predictions = predict_temperature_for_week(date_input)

    prediction_results = pd.DataFrame({
        'DATE': [date for date, _ in weekly_predictions],
        'Predicted TAVG': [temp for _, temp in weekly_predictions]
    })

    # Pass the DataFrame directly to the template
    return render_template('results.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
