# Class to test the model

from DataPreparation_graph import DataPreparation as dt
import ModelService as model

# Set the model name
model_name="graph-3mo-1v-96h"

# Instantiate the class
classModule = dt.DataPreparation(grid_step=0.22)

# Train-test split
adj_matrix_norm_train, adj_matrix_norm_test, adj_matrix_norm_validation, sample_train, target_train, sample_test, target_test, sample_validation, target_validation = classModule.prepareDataForGraphModel(
                                                                       start_date="2025-08-01",
                                                                       end_date="2025-10-07",
                                                                       variableToPredict=["temperature"],
                                                                       test_size=0.30,
                                                                       validation_size=0.15,
                                                                       window_size=24,
                                                                       horizon=96,
                                                                       distance_threshold=10,
                                                                       save_name=model_name)
# Launch model
model.ModelService(train_set=sample_train,
                   train_labels=target_train,
                   test_set=sample_test,
                   test_labels=target_test,
                   validation_set=sample_validation,
                   validation_labels=target_validation).WaveNetTimeSpaceModel(adj_matrix_train=adj_matrix_norm_train,
                                                                              adj_matrix_test=adj_matrix_norm_test,
                                                                              model_params={"channels_t": 32,
                                                                                            "channels_s": 32,
                                                                                            "dilations": (12, 24),
                                                                                            "kernel_size": 2},
                                                                              training_epochs=3,
                                                                              save_name=model_name,
                                                                              # The following are only valid for prediction's sake
                                                                              variableToPredict=["temperature"],
                                                                              end_date="2025-10-07"
                                                                              )

