import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

def create_date_features(date):
    return pd.Series({
        'day_of_year': date.dayofyear,
        'day_of_month': date.day,
        'month': date.month,
        'year': date.year,
        'is_weekend': date.weekday() >= 5,
    })

# Load and preprocess the data
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
    start_date = pd.to_datetime(start_date, format='%d-%m-%Y')
    predictions = []

    for i in range(7):
        current_date = start_date + timedelta(days=i)
        input_features = create_date_features(current_date).to_frame().T
        
        # Use historical averages for weather features
        historical_data = weather[weather.index.month == current_date.month]
        input_features['PRCP'] = historical_data['PRCP'].mean()
        input_features['TMAX'] = historical_data['TMAX'].mean()
        input_features['TMIN'] = historical_data['TMIN'].mean()
        
        imputed_input = imputer.transform(input_features[features.columns])
        scaled_input = scaler.transform(imputed_input)
        prediction = model.predict(scaled_input)[0]
        
        predictions.append((current_date, prediction))

    return predictions

# Example usage
while True:
    input_date = input("Enter a start date for the week (DD-MM-YYYY) or 'q' to quit: ")
    if input_date.lower() == 'q':
        break
    
    weekly_predictions = predict_temperature_for_week(input_date)
    print(f"\nPredicted temperatures for the week starting {input_date}:")
    for date, temp in weekly_predictions:
        print(f"{date.strftime('%d-%m-%Y')}: {temp:.2f}°C")
    print()