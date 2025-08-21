import DataPreparation as dt
import numpy as np
import time

start = time.time()

# Instantiate the class
classModule = dt.DataPreparation(grid_step=0.22)

# Train-test split
adj_matrix_norm_train, adj_matrix_norm_test, adj_matrix_norm_validation, sample_train, target_train, sample_test, target_test, sample_validation, target_validation = classModule.prepareDataForGraphModel(
                                                                       start_date = "2025-06-01",
                                                                       end_date = "2025-08-15",
                                                                       variableToPredict=["temperature"],
                                                                       test_size=0.30,
                                                                       validation_size=0.15,
                                                                       window_size=24,
                                                                       horizon=96,
                                                                       distance_threshold=50)

end = time.time()
print("\nTime Elapsed for dataset preparation: " + str(round((end-start) / 60, 2)) + " minutes")