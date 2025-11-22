# Class to test the model

from ModelService_graph.CloudService import CloudModelServiceWrapper as cmodel

# Prepare data + train + save the model params
cmodel.CloudModelServiceWrapper(model_name="graph-3mo-1v-96h-temperature-small",
                                variableToPredict=["temperature"]).trainAndSaveGraphModel(model_params={"channels_t": 32,
                                                                                                  "channels_s": 32,
                                                                                                  "dilations": (1, 2),
                                                                                                  "kernel_size": 2},
                                                                                          test_size=0.30,
                                                                                          validation_size=0.15,
                                                                                          window_size=24,
                                                                                          prediction_horizon=96,
                                                                                          matrix_params={"type": "KNN",  # "KNN" | "distance"
                                                                                                         "threshold": 300},
                                                                                          training_epochs=3,
                                                                                          environment="local"
                                                                                          )