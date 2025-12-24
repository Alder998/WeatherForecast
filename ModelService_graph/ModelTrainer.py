# Class to test the model

from ModelService_graph import ModelServiceWrapper as model

# Prepare data + train + save the model params (min. time_steps: time_steps - window_size - horizon + 1, for every sample possible)
model.ModelServiceWrapper(grid_step=0.22,
                          model_name="graph-1mo-5v-96h-wmean",
                          variableToPredict=["temperature", "humidity_mean", "windSpeed", "cloudCover", "pressure_msl"]).trainAndSaveGraphModel(model_params={"channels_t": 32,
                                                                                                  "channels_s": 32,
                                                                                                  "dilations": [12, 24],
                                                                                                  "kernel_size": 2},
                                                                                    start_date="2025-11-01",
                                                                                    end_date="2025-12-20",
                                                                                    test_size=0.30,
                                                                                    validation_size=0.15,
                                                                                    window_size=24,
                                                                                    prediction_horizon=96,  # default: 96
                                                                                    matrix_params={"type": "KNN",  # "KNN" | "distance"
                                                                                                   "threshold": 300},
                                                                                    training_epochs=10,
                                                                                    stabilize_trend=True,
                                                                                    environment="local"
                                                                                    )