# Class to continue Model Training after checkpoint more easily

from ModelService_graph import ModelServiceWrapper as model

# Prepare data + train + save the model params
model.ModelServiceWrapper(grid_step=0.22,
                          model_name="graph-3mo-1v-96h",
                          variableToPredict=["cloudCover"]).continueModelTraining(
                                                                                    start_date="2025-08-01",
                                                                                    end_date="2025-10-07",
                                                                                    split_method="time", # "time-space" | "time"
                                                                                    test_size=0.30,
                                                                                    validation_size=0.15,
                                                                                    window_size=24,
                                                                                    prediction_horizon=96,
                                                                                    matrix_params={"type": "KNN",  # "KNN" | "distance"
                                                                                                   "threshold": 300},
                                                                                    new_epochs=3
                                                                                    )