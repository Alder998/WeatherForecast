# Class to test the model

from DataPreparation_graph import DataPreparation as dt
import ModelService as model

# Instantiate the class
classModule = dt.DataPreparation(grid_step=0.22)

# Train-test split
adj_matrix_norm_train, adj_matrix_norm_test, adj_matrix_norm_validation, sample_train, target_train, sample_test, target_test, sample_validation, target_validation = classModule.prepareDataForGraphModel(
                                                                       start_date = "2024-08-01",
                                                                       end_date = "2025-08-15",
                                                                       variableToPredict=["temperature"],
                                                                       test_size=0.30,
                                                                       validation_size=0.15,
                                                                       window_size=24,
                                                                       horizon=96,
                                                                       distance_threshold=10)
# Launch model
model.ModelService(train_set=sample_train,
                   train_labels=target_train,
                   test_set=sample_test,
                   test_labels=target_test,
                   validation_set=sample_validation,
                   validation_labels=target_validation).WaveNetTimeSpaceModel(adj_matrix_train=adj_matrix_norm_train,
                                                                              training_epochs=2)

