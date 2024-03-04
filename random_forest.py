import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor

json_file_path = '/Users/fanjian/Desktop/final_project/stage3_features_v2.json'
new_data_json_path = '/Users/fanjian/Desktop/final_project/future_data_1218~1224.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

with open(new_data_json_path, 'r') as file:
    new_data = json.load(file)

df_list = []
for station_id, station_data in data.items():
    for date, date_data in station_data.items():
        for time, time_data in date_data.items():
            df_list.append({
                'date': date,
                'station_id': station_id,
                'time': time,
                'tot': time_data['tot'],
                'sbi': time_data['sbi'],
                'bemp': time_data['bemp'],
                'act': time_data['act'],
                'K': time_data['K'],
                'event': time_data['event'],
                'Holiday': time_data['isHoliday'],
                'Distance': time_data['Distance'],
                'Humidity': time_data['Humidity'],
                'Temperature': time_data['Temperature'],
                'Peak': time_data['Peak']
            })
df = pd.DataFrame(df_list)

new_df_list = []
for station_id, station_data in new_data.items():
    for date, date_data in station_data.items():
        for time, time_data in date_data.items():
            new_df_list.append({
                'date': date,
                'station_id': station_id,
                'time': time,
                'tot': time_data['tot'],
                'Peak': time_data['Peak'],
                'Distance': time_data['Distance'],
                'K': time_data['K'],
                'Holiday': time_data['isHoliday'],
                'Temperature': time_data['Temperature'],
                'Humidity': time_data['Humidity'],
                'event': time_data['event']
            })
new_df = pd.DataFrame(new_df_list)

features = ['time', 'tot', 'Peak', 'K', 'Holiday']
target = 'sbi'

new_data_processed = new_df[features]

X = df[features]
y = df[target]

X_train, y_train = X, y

model = RandomForestRegressor()
model.fit(X_train, y_train)

new_predictions = model.predict(new_data_processed)
rounded_predictions = new_predictions.round().astype(int)
predictions_df = pd.DataFrame({'date': new_df['date'],
                               'station_id': new_df['station_id'],
                               'time': new_df['time'],
                               'Predicted_sbi': rounded_predictions})

predictions_df.sort_values(by=['date', 'station_id'], inplace=True)

csv_file_path = '/Users/fanjian/Desktop/final_project/predictions.csv'

predictions_df.to_csv(csv_file_path, index=False)
