# Class to test the model

from ModelService_graph import ModelServiceWrapper as model

# Prepare data + train + save the model params
model.ModelServiceWrapper(grid_step=0.22,
                          model_name="graph-3mo-1v-96h",
                          variableToPredict=["temperature"]).trainAndSaveGraphModel(model_params={"channels_t": 32,
                                                                                                               "channels_s": 32,
                                                                                                               "dilations": (12, 24),
                                                                                                               "kernel_size": 2},
                                                                                                 start_date="2025-08-01",
                                                                                                 end_date="2025-10-07",
                                                                                                 test_size=0.30,
                                                                                                 validation_size=0.15,
                                                                                                 window_size=24,
                                                                                                 prediction_horizon=96,
                                                                                                 distance_threshold=150,
                                                                                                 training_epochs=3
                                                                                                 )