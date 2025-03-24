# Class to test the model

from DataPreparation import DataPreparation as dt
import ModelService as model

# Get the data
# Available for model: 'temperature' | 'precipitation' | 'humidity_mean' | 'windSpeed' | 'cloudCover' | 'pressure_msl'
train_set, test_set, train_labels, test_labels = dt.DataPreparation(grid_step=0.22).getDataForModel(start_date='2025-01-01',
                                                   end_date='2025-01-15',
                                                   test_size=0.20,
                                                   predictiveVariables=['date', 'latitude', 'longitude'],
                                                   variableToPredict='temperature')
# Train the Model
# Model Structure
structure = {'FF': [500, 500, 500, 500, 500],
             'LSTM': [50, 50, 50, 50],
             'Conv': []}
model.ModelService(train_set, test_set, train_labels, test_labels).NNModel(modelStructure=structure,
                                                                           trainingEpochs=100,
                                                                           return_seq_last_rec_layer=False)





