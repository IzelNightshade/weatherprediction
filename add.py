import pandas as pd

weather = pd.read_csv('cleaned.csv')

date = '2023-11-14'
station = 'AEM00041194'
name = 'DUBAI INTERNATIONAL, AE'
precipitation = 0.0
avgtemp = 25.2


dict = {'DATE':pd.to_datetime(date).strftime('%d-%m-%Y'), 'STATION':station, 'NAME':name, 'PRCP':precipitation, 'TAVG':avgtemp}
new_row = pd.DataFrame(dict, index=[0])

# Append row to the dataframe
weather = pd.concat([weather,new_row],ignore_index=True)


print(weather)

weather.to_csv('cleaned.csv', index=False) 