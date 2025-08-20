import DataPreparation as dt
import numpy as np

# Instantiate the class
classModule = dt.DataPreparation(grid_step=0.22)

# Train-test split
adj_matrix_norm, feature_matrix = classModule.prepareDataForGraphModel(start_date = "2025-06-01",
                                                                       end_date = "2025-06-20",
                                                                       variableToPredict=["temperature"])
