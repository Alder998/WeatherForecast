# Class that allow to make prediction on future data and visualize them so to understand if the model has scored well
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from DataPreparation import DataPreparation as dt
import tensorflow as tf
import ModelService as model

# Set silent option on downcasting to avoid warning
pd.set_option('future.no_silent_downcasting', True)

class PredictionService:

    def __init__(self, model, grid_step, start_date, prediction_steps):
        self.model = model
        self.grid_step = grid_step
        self.start_date = start_date
        self.prediction_steps = prediction_steps
        pass

    # Utils-like function to standardize the data
    def standardizeData(self, set):

        # Initialize the scaler from scikit-learn
        scaler = StandardScaler()

        # Reshape that is necessary to process the features separately
        num_samples, num_obs, num_features = set.shape
        X_reshaped = set.reshape(-1, num_features)

        # Normalize the features
        X_scaled = scaler.fit_transform(X_reshaped)

        # Put the array in the old shape
        X_scaled = X_scaled.reshape(num_samples, num_obs, num_features)

        return X_scaled, scaler

    def dataClass (self, type='db'):

        env_path = r"D:\PythonProjects-Storage\WeatherForecast\App_core\app.env"
        load_dotenv(env_path)
        database = os.getenv("database")
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")

        # Instantiate the database Object
        if type == 'db':
            dataClass = db.Database(database, user, password, host, port)
        elif type == 'dk':
            dataClass = dk.Database_dask(database, user, password, host, port)
        else:
            raise Exception('The DB type ' + type + ' has been mispelled or it has not be implemented!')

        return dataClass

    # We want to make prediction ahead in time, therefore we must create a dataset in the same
    # shape as the model's one (using the space grid), and predict on that
    def createPredictionSet (self):

        # import the time grid
        gridPointData = self.dataClass().getDataFromTable("gridPoints_" + str(self.grid_step)).drop_duplicates()

        # Now, from the start date, create an hourly frame with the length of the prediction steps
        start_date = pd.to_datetime(self.start_date) + pd.Timedelta(days=1)
        predictionData = pd.DataFrame(pd.date_range(start=start_date, periods=self.prediction_steps,
                                                    freq='h')).set_axis(['date'], axis = 1)

        rawPredictionSet = []
        for gridPoint in range(len(gridPointData[gridPointData.columns[0]])):
            # Isolate the single couples of coordinates, concatenate them with the grid part, front-fill the NaN
            # Create the grid part isolating the couples of coordinates
            gridPart = gridPointData.iloc[gridPoint].reset_index().T
            gridPart = gridPart.set_axis([gridPart.loc['index']], axis = 1)[1:].reset_index(drop=True)
            gridPart.columns = gridPart.columns.get_level_values(0)
            # Now concatenate and front-fill
            dateWithGridCouple = pd.concat([predictionData, gridPart], axis = 1)
            dateWithGridCouple = dateWithGridCouple.ffill().infer_objects(copy=False)
            rawPredictionSet.append(dateWithGridCouple)
        rawPredictionSet = pd.concat([df for df in rawPredictionSet], axis = 0)

        # Now, adapt the data for model with the appropriate function
        predictionSet = dt.DataPreparation(self.grid_step).adaptDataForModel(rawPredictionSet,
                                        ['year','month','day','hour','lat','lng'])

        return predictionSet

    # Class to load the model and run the predictions
    def NNPredict (self):

        # Load the NN Model
        model = tf.keras.models.load_model(self.model + '.h5')

        # Load the data creating a time-ahead set
        predictionSetRaw = self.createPredictionSet()
        # Adapt data for Model
        predictionSet = np.stack(predictionSetRaw, axis=0)

        # Standardize data
        predictionSet, used_scaler = self.standardizeData(predictionSet)

        # Visualize the data size
        print('Prediction Set Size: ', self.getSetSize(predictionSet))

        # Make the Prediction
        predictions = model.predict(predictionSet)

        # TODO: implement a method to do the inverse transformation and de-standardize the data
        # Create the DataFrame for the prediction
        predictions_reshaped = predictions.reshape(-1, 1)
        #predictions_reshaped_df = pd.DataFrame(predictions_reshaped, columns=['prediction'])


        return predictions

    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck









