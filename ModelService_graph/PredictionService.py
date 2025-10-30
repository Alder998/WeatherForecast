# Class that allow to make prediction on future data and visualize them so to understand if the model has scored well
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from prophet import Prophet
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# Add all folders for batch execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from DataPreparation_graph import DataPreparation as dt
import tensorflow as tf
from ModelService_graph.TensorFlowService import GraphWaveNet as gwn
import joblib
import json

# Set silent option on downcasting to avoid warning
pd.set_option('future.no_silent_downcasting', True)

class PredictionService:

    def __init__(self, model):
        self.model = model
        pass

    # function to de-standardize the data with loaded scaler
    def deStandardizeData(self, data, loaded_scaler):
        # Take the set shape
        B, N, F, T = data.shape

        # Use the same shape as of standardization
        data_reshaped = data.transpose(0, 1, 3, 2).reshape(-1, F)

        # Do the inverse transform
        data_descaled = loaded_scaler.inverse_transform(data_reshaped)

        # Put the data in teh original shape
        data_descaled = data_descaled.reshape(B, N, T, F).transpose(0, 1, 3, 2)

        return data_descaled

    def dataClass(self, type='db'):

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

        return dataClass

    # Main function to prepare data for the selected Model to be predicted
    def predictWithStoredModel(self, grid_step=0.22, start_date=""):

        # 1. Load the necessary Inputs

        # 1.1. Load the Adjacency Matrix
        loaded_adjMatrix = np.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\AdjacencyMatrix.npy")
        print("DATA PREPARATION FOR PREDICTION - Shape of loaded Adjacency Matrix: ", loaded_adjMatrix.shape)

        # 1.2. Load the Model
        # 1.2.1. Load the config
        print("DATA PREPARATION FOR PREDICTION - Loading Model Weights and config...")
        with open("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\model_config.h5", "r") as f:
            config = json.load(f)
        # 1.2.2. Load the model and build it
        model = gwn.GraphWaveNet(N=config["model_params"]["N"], F_in=config["model_params"]["F_in"], W=config["model_params"]["W"],
                                 H=config["model_params"]["H"], n_blocks=config["model_params"]["n_blocks"],
                                 channels_s=config["model_user_params"]["channels_s"],
                                 channels_t=config["model_user_params"]["channels_t"],
                                 dilations=config["model_user_params"]["dilations"],
                                 kernel_size=config["model_user_params"]["kernel_size"],
                                 A=loaded_adjMatrix).build_graph_wavenet()
        # 1.2.3. Load the weights
        model.load_weights("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\model_weights.weights.h5")

        # 1.3. Load the last Layer
        loaded_last_layer = np.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\last_obs_layer.npy")
        print("DATA PREPARATION FOR PREDICTION - Shape of loaded Last Layer (for prediction shape): ", loaded_last_layer.shape)

        # 2. predict based on last layer
        modelPrediction = model.predict(loaded_last_layer)
        print("PREDICTION: size of prediction:", modelPrediction.shape)
        # 2.1 load the scaler
        loaded_scaler = joblib.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\scaler.pkl")
        # 2.2. De-scale the prediction
        modelPrediction = self.deStandardizeData(data=modelPrediction, loaded_scaler=loaded_scaler)
        print("PREDICTION: size of de-scaled prediction:", modelPrediction.shape)

        # 3. Put the prediction into a readable output
        # 3.1. Take all the admissible coordinates from the database
        allCoords = self.dataClass().executeQuery('SELECT * FROM public."gridPoints_' + str(grid_step) + '"')
        uniqueCoords = allCoords.drop_duplicates()
        # 3.2. create a series of dates to start the prediction from
        dates = pd.date_range(start = datetime.strptime(start_date, "%Y-%m-%d"), periods=modelPrediction.shape[3], freq="h")

        prediction_dataset = []
        for point_step in range(modelPrediction.shape[2]):
            prediction_for_point = pd.DataFrame(np.squeeze(modelPrediction[:, :, point_step, :], axis=0))
            # Set index and columns appropriately: index must refer to the coordinates, so take them from database
            prediction_df_format = pd.concat([uniqueCoords, prediction_for_point.set_axis(dates, axis=1)], axis=1)
            dataset_for_representation = []
            for d in dates:
                data_sql_format = pd.concat([pd.DataFrame(np.full(len(prediction_df_format[prediction_df_format.columns[0]]), d)),
                                             prediction_df_format[["lat", "lng"]],
                                             prediction_df_format[d]], axis=1).set_axis(["date", "latitude",
                                                                                           "longitude", config["variableToPredict"][point_step]], axis=1)
                dataset_for_representation.append(data_sql_format)
            dataset_for_representation = pd.concat([df for df in dataset_for_representation], axis = 0).reset_index(drop=True)
            prediction_dataset.append(dataset_for_representation)

        if len(prediction_dataset) == 1:
            return prediction_dataset[0]
        else:
            prediction_dataset = pd.concat([df for df in prediction_dataset], axis=1)
            # Drop duplicate columns (ideally, latitude, longitude, date) and return
            prediction_dataset = prediction_dataset.loc[:, ~prediction_dataset.columns.duplicated()]
            return prediction_dataset

