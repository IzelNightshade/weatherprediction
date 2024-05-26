import pandas as pd

weather = pd.read_csv('cleaned.csv')

date = '2023-11-09'
station = 'AEM00041194'
name = 'DUBAI INTERNATIONAL, AE'
precipitation = 0.0
avgtemp = 0.0

new_row = {'DATE': pd.to_datetime(date) , 'STATION': station, 'NAME': name, 'PRCP': precipitation, 'TAVG': avgtemp}



# weather = weather.append(new_row, ignore_index=True)

print(weather)