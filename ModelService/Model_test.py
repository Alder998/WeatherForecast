# Class to test the model
from datetime import datetime
from DataPreparation import DataPreparation as dt
import ModelService as model

# Variables to fill the model and the model name
variableToPredict = 'temperature'
start_date = '2025-02-01'
end_date = '2025-03-31'
trainingEpochs = 50
timeSpan = (datetime.strptime(end_date, '%Y-%m-%d') -
              datetime.strptime(start_date, '%Y-%m-%d')).days
print('Time span for Model: ' + str(timeSpan) + 'd')

# Get the data
# Available for model: 'temperature' | 'precipitation' | 'humidity_mean' | 'windSpeed' | 'cloudCover' | 'pressure_msl'
train_set, test_set, train_labels, test_labels = dt.DataPreparation(grid_step=0.22).getDataForModel(
                                                   start_date=start_date,
                                                   end_date=end_date,
                                                   test_size=0.20,
                                                   predictiveVariables=['year', 'month', 'day', 'hour', 'latitude',
                                                                        'longitude', 'windSpeed', 'pressure_msl', 'humidity_mean'],
                                                   variableToPredict=variableToPredict,
                                                   time_split=True,
                                                   space_split=True)
# Train the Model
# Model Structure
structure = {'FF': [500, 500, 500, 500, 500],
             'LSTM': [64, 64, 64, 64],
             'Conv': [64, 64, 64, 64]}
model.ModelService(train_set, test_set, train_labels, test_labels).NNModel(modelStructure=structure,
                                                                           trainingEpochs=trainingEpochs,
                                                                           standardize=True,
                                                                           return_seq_last_rec_layer=False,
                                                                           save_name='WeatherForecastModel_TimeSpaceSplit_' +
                                                                variableToPredict + '_' + str(timeSpan) +
                                                                'd_' + str(trainingEpochs) + 'Epochs_WaddingVariables')