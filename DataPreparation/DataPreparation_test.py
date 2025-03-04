import DataPreparation as dt

# get the data from database

grid_step = 0.22

data = dt.DataPreparation(tableName = 'WeatherForRegion_'+ str(grid_step),
                   predictiveVariables = ['latitude', 'longitude'],
                   variableToPredict = 'precipitation').adaptDataForModel(time_steps = 1000)

print(data)