# Class to test the model

import os
import sys

# Add all folders for batch execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from DataPreparation import DataPreparation as dt
import ModelService as model

modelName = 'small-bidirectional-LSMT-uniform_time'

# Variables to fill the model and the model name
variableToPredict = 'temperature_residual'
start_date = '2024-02-01'
end_date = '2025-07-11'
trainingEpochs = 3
timeSplit = True
spaceSplit = True
nearPointsPerGroup = 30
test_size = 0.20
# flag to take an existing Model, and continue the training
continue_training = {'continue': False,
                     'new_epochs': 3}

# Set the model name
timeSpan = (datetime.strptime(end_date, '%Y-%m-%d') -
            datetime.strptime(start_date, '%Y-%m-%d')).days
print('Time span for Model: ' + str(timeSpan) + 'd')

# Now, get the data from Model
# Get the data
# Available for model: 'temperature' | 'precipitation' | 'humidity_mean' | 'windSpeed' | 'cloudCover' | 'pressure_msl'
train_set, test_set, train_labels, test_labels = dt.DataPreparation(grid_step=0.22).getDataForModel(
                                                   start_date=start_date,
                                                   end_date=end_date,
                                                   test_size=test_size,
                                                   predictiveVariables=["year", "month", "day", "hour", "latitude",
                                                                        "longitude", "seasonal"],
                                                   timeVariables=["year", "month", "day", "hour"],
                                                   variableToPredict=variableToPredict,
                                                   time_split=timeSplit,
                                                   space_split=spaceSplit,
                                                   nearPointsPerGroup=nearPointsPerGroup,
                                                   space_split_method = "radius",
                                                   plot_space_split=False,
                                                   modelName=modelName,
                                                   time_split_method="sparse_weekly",
                                                   seasonal_decomposition=True)

if not continue_training['continue']:

    # Train the Model
    # Model Structure
    structure = {'FF': [500, 500],
                 'LSTM': [64, 64, 64],
                 'Conv1D': [64, 64, 64],
                 'Conv2D': [],
                 'Conv2DLSTM': []}
    model.ModelService(train_set, test_set, train_labels, test_labels).NNModel(modelStructure=structure,
                                                                               trainingEpochs=trainingEpochs,
                                                                               dropout_FF=0.20,
                                                                               dropout_LSTM=0.20,
                                                                               standardize=True,
                                                                               return_seq_last_rec_layer=False,
                                                                               save_name=modelName)
else:
    # Load model, continue training
    model.ModelService(train_set, test_set, train_labels, test_labels).continueModelTraining(modelName=modelName,
                                                                                             newTrainingEpochs=continue_training['new_epochs'])
