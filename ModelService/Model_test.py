# Class to test the model

from DataPreparation import DataPreparation as dt
import ModelService as model

# Get the data
train_set, test_set, train_labels, test_labels = dt.DataPreparation(grid_step=0.22).getDataForModel(start_date='2025-01-01',
                                                   end_date='2025-01-10',
                                                   test_size=0.20,
                                                   predictiveVariables=['date', 'latitude', 'longitude'],
                                                   variableToPredict='precipitation')
# Train the Model
# Model Structure
structure = {'FF': [500, 500, 500, 500, 500],
             'LSTM': [30, 30, 30]}
model.ModelService(train_set, test_set, train_labels, test_labels).NNModel(modelStructure=structure,
                                                                           trainingEpochs=30)





