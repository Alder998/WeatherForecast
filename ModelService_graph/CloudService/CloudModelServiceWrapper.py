"""
This class will be maily used to organize better model inputs for the launch's sake
"""

from DataPreparation_graph import DataPreparation as dt
from ModelService_graph import ModelService as model

class CloudModelServiceWrapper:
    def __init__(self):
        pass

    def trainAndSaveGraphModelOnCloudMachine (self, model_name, variableToPredict, model_params, test_size, validation_size, window_size, prediction_horizon,
                                matrix_params, training_epochs, environment, start_date, end_date):

        # Instantiate the class
        classModule = dt.DataPreparation(grid_step="", environment=environment)

        # Train-test split
        adj_matrix_norm, sample_train, target_train, sample_test, target_test, sample_validation, target_validation = classModule.prepareDataForGraphModel(
            start_date=start_date,
            end_date=end_date,
            variableToPredict=variableToPredict,
            test_size=test_size,
            validation_size=validation_size,
            window_size=window_size,
            horizon=prediction_horizon,
            matrix_params=matrix_params,
            save_name=model_name
        )

        model.ModelService(train_set=sample_train,
                           train_labels=target_train,
                           test_set=sample_test,
                           test_labels=target_test,
                           validation_set=sample_validation,
                           validation_labels=target_validation,
                           environment=environment).WaveNetTimeSpaceModel(
                                                                          adj_matrix=adj_matrix_norm,
                                                                          model_params=model_params,
                                                                          training_epochs=training_epochs,
                                                                          save_name=model_name,
                                                                          variableToPredict=variableToPredict,
                                                                          end_date=""
                                                                          )

    # Method to continue the Model training loading the existing model
    def continueModelTraining (self, variableToPredict, model_name, test_size, validation_size, window_size,
                               prediction_horizon, matrix_params, new_epochs, environment):

        # The data prep class is always required
        # Instantiate the class
        classModule = dt.DataPreparation(grid_step="", environment=environment)

        # Train-test split
        adj_matrix_norm, sample_train, target_train, sample_test, target_test, sample_validation, target_validation = classModule.prepareDataForGraphModel(
            start_date="",
            end_date="",
            variableToPredict=variableToPredict,
            test_size=test_size,
            validation_size=validation_size,
            window_size=window_size,
            horizon=prediction_horizon,
            matrix_params=matrix_params,
            save_name=model_name)

        # Ad-hoc method within modelService to load and continue model training
        model.ModelService(train_set=sample_train,
                           train_labels=target_train,
                           test_set=sample_test,
                           test_labels=target_test,
                           validation_set=sample_validation,
                           validation_labels=target_validation,
                           environment=environment).continueModelTraining(model_name=model_name,
                                                                          new_epochs=new_epochs)