"""
Class to manage the storage of models only providing the model name
This must be functional while loading the model for prediction
"""

class ModelStorageService:

    # The model Name must be the only info provided
    def __init__(self, modelName):
        self.modelName = modelName
        pass

    # Desired structure of stored Models: a directory with the Model name (given by the user)
    # That contain all the possible model infos
    def saveModel (self, epochs, NN_structure, predictive_variables, test_loss, geo_split, trainTest_split_size):

        # The info needed to store the model are:
        # - Training Epochs
        # - NN structure
        # - Predictive variables used
        # - Test loss
        # - Train-Test
        # -- Geo split features
        # -- Overall train and test shape



        return 0
