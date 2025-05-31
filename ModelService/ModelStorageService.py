import json

"""
Class to manage the storage of models only providing the model name
This must be functional while loading the model for prediction
The info needed to store the model are:
- Training Epochs (Model)
- NN structure (Model)
- Predictive variables used (Data)
- Test loss (Model)
- Train-Test
-- Geo split features (Data)
-- Overall train and test shape (Data)
"""

class ModelStorageService:

    # The model Name must be the only info provided
    def __init__(self, modelName, model):
        self.model = model
        self.modelName = modelName
        pass

    # Desired structure of stored Models: a directory with the Model name (given by the user)
    # That contain all the possible model infos from the "data" part
    def saveModelDataInfo (self, predictive_variables, split_method, geo_split, train_shape, test_shape, trainTest_split_perc):

        # The fastest way to store them could be a JSON

        # create the basic JSON
        modelInfo = {}
        # Save the features that are directly from the model module
        modelInfo['predictive_variables'] = predictive_variables  # List
        modelInfo['split_method'] = split_method  # Time-Split | Time-Space-Split | Space-Split
        modelInfo['geo_split'] = geo_split  # Uniform | Radius | Other methods to space split the data
        modelInfo['train_shape'] = train_shape  # Features in shape (timeFrame, spaceFrame, features)
        modelInfo['test_shape'] = test_shape  # Features in shape (timeFrame, spaceFrame, features)
        modelInfo['trainTest_split_perc'] = trainTest_split_perc  # %, overall

        # Convert everything into string
        data_str_keys = {str(k): v for k, v in modelInfo.items()}

        # Save into the stored models directory
        with open("modelInfo", "w") as f:
            json.dump(data_str_keys, f, indent=4)

        return modelInfo
