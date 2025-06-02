import json
import os

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
    def __init__(self, modelName):
        self.modelName = modelName
        pass

    # Desired structure of stored Models: a directory with the Model name (given by the user)
    # That contain all the possible model infos from the "data" part
    def saveModelDataInfo (self, grid_step, predictive_variables, split_method, geo_split, target_variable, timeVariables):

        # create directory with the model name
        folder_path = "D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.modelName

        if os.path.exists(folder_path + '\\modelInfo.json'):
            # If exists, read the JSON, update it
            with open(folder_path + '\\modelInfo.json', "r") as f:
                modelInfo = json.load(f)
        else:
            # create the basic JSON
            os.makedirs(folder_path, exist_ok=True)
            modelInfo = {}

        # Save the features that are directly from the model module
        modelInfo['predictive_variables'] = predictive_variables  # List
        modelInfo['split_method'] = split_method  # Time-Split | Time-Space-Split | Space-Split
        modelInfo['geo_split'] = geo_split  # Uniform | Radius | Other methods to space split the data
        modelInfo['target_variable'] = target_variable  # target variable
        modelInfo['grid_step'] = grid_step  # grid step
        modelInfo['time_variables'] = timeVariables # Time variables

        # Convert everything into string
        data_str_keys = {str(k): v for k, v in modelInfo.items()}

        # Save into the stored models directory
        with open(folder_path + "\\modelInfo.json", "w") as f:
            json.dump(data_str_keys, f, indent=None)
            print('Model data Info saved correctly!')

        return modelInfo

    # Funtion to store Model info
    def saveModelInfo (self, NN_structure, epochs, training_test_shape, test_set_shape, test_loss):

        # This function will always run after the function above. Therefore, a JSON will always be present
        folder_path = "D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.modelName + '\\modelInfo.json'
        with open(folder_path, "r") as f:
            existingJSON = json.load(f)

        # Fill the info JSON with the remaining details
        existingJSON['NN_Structure'] = NN_structure
        existingJSON['Training Epochs'] = epochs
        existingJSON['Train set shape'] = training_test_shape
        existingJSON['Test set shape'] = test_set_shape
        existingJSON['Test loss'] = test_loss

        # Convert everything into string
        data_str_keys = {str(k): v for k, v in existingJSON.items()}

        # Save into the stored models directory
        with open(folder_path, "w") as f:
            json.dump(data_str_keys, f, indent=None)
            print('Model Info saved correctly!')

        return existingJSON
