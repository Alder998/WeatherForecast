# Class to test the model

from DataPreparation import DataPreparation as dt
import ModelService as model

# Get the data
# Available for model: 'temperature' | 'precipitation' | 'humidity_mean' | 'windSpeed' | 'cloudCover' | 'pressure_msl'
train_set, test_set, train_labels, test_labels = dt.DataPreparation(grid_step=0.22).getDataForModel(start_date='2025-01-01',
                                                   end_date='2025-01-30',
                                                   test_size=0.20,
                                                   predictiveVariables=['year', 'month', 'day', 'hour', 'latitude', 'longitude'],
                                                   variableToPredict='temperature',
                                                   time_split=True,
                                                   space_split=True)
# Train the Model
# Model Structure
structure = {'FF': [500, 500, 500, 500, 500],
             'LSTM': [64, 64, 64, 64],
             'Conv': [64, 64, 64, 64]}
model.ModelService(train_set, test_set, train_labels, test_labels).NNModel(modelStructure=structure,
                                                                           trainingEpochs=100,
                                                                           standardize=True,
                                                                           return_seq_last_rec_layer=False,
                                                                           save_name='WeatherForecastModel_geoTimeSplit_test')