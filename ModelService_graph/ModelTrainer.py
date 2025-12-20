# Class to test the model

from ModelService_graph import ModelServiceWrapper as model

# Prepare data + train + save the model params (min. time_steps: time_steps - window_size - horizon + 1, for every sample possible)
model.ModelServiceWrapper(grid_step=0.22,
                          model_name="graph-3mo-1v-96h-temperature-small",
                          variableToPredict=["temperature", "temperature_trend"]).trainAndSaveGraphModel(model_params={"channels_t": 32,
                                                                                                  "channels_s": 32,
                                                                                                  "dilations": [24, 48, 168, 720],
                                                                                                  "kernel_size": 2},
                                                                                    start_date="2025-09-01",
                                                                                    end_date="2025-10-07",
                                                                                    test_size=0.30,
                                                                                    validation_size=0.15,
                                                                                    window_size=24,
                                                                                    prediction_horizon=48,  # default: 96
                                                                                    matrix_params={"type": "KNN",  # "KNN" | "distance"
                                                                                                   "threshold": 300},
                                                                                    training_epochs=3,
                                                                                    environment="local"
                                                                                    )