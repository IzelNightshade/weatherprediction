import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


weather = pd.read_csv('cleaned.csv')


# Specify the column to be set as the new index (replace 'DATW' with the actual column name)
new_index_column = 'DATE'

# Convert the 'DATE' column to datetime format using the specified format 'dd-mm-yyyy'
weather[new_index_column] = pd.to_datetime(weather[new_index_column], format='%d-%m-%Y')

# Set the 'DATE' column as the new index
weather.set_index(new_index_column, inplace=True)
weather[['TAVG']] = weather[['TAVG']] / 10

weather["target"] = weather.shift(-1)["TAVG"]
weather = weather.ffill()


rr = Ridge(alpha=.1)


predictors = weather.columns[~weather.columns.isin(["target", "NAME", "STATION"])]


def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)

predictions = backtest(weather, rr, predictors)

print(predictions.head)



mae = mean_absolute_error(predictions["actual"], predictions["prediction"])
print("MAE: ", mae)