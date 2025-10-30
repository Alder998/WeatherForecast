"""
This class will be maily used to organize better model inputs for the launch's sake
"""

from DataPreparation_graph import DataPreparation as dt
from ModelService_graph import ModelService as model

class ModelServiceWrapper:
    def __init__(self, model_name, grid_step, variableToPredict):
        self.model_name = model_name
        self.grid_step = grid_step
        self.variableToPredict = variableToPredict
        pass

    def trainAndSaveGraphModel (self, model_params, start_date, end_date, test_size, validation_size, window_size, prediction_horizon,
                                distance_threshold, training_epochs):

        # Instantiate the class
        classModule = dt.DataPreparation(grid_step=self.grid_step)

        # Train-test split
        adj_matrix_norm, sample_train, target_train, sample_test, target_test, sample_validation, target_validation = classModule.prepareDataForGraphModel(
            start_date=start_date,
            end_date=end_date,
            variableToPredict=self.variableToPredict,
            test_size=test_size,
            validation_size=validation_size,
            window_size=window_size,
            horizon=prediction_horizon,
            distance_threshold=distance_threshold,
            save_name=self.model_name)

        model.ModelService(train_set=sample_train,
                           train_labels=target_train,
                           test_set=sample_test,
                           test_labels=target_test,
                           validation_set=sample_validation,
                           validation_labels=target_validation).WaveNetTimeSpaceModel(
                                                                                    adj_matrix=adj_matrix_norm,
                                                                                    model_params=model_params,
                                                                                    training_epochs=training_epochs,
                                                                                    save_name=self.model_name,
                                                                                    # The following are only valid for prediction's sake
                                                                                    variableToPredict=self.variableToPredict,
                                                                                    end_date=end_date
                                                                                    )