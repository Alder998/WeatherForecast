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

    # Main function to prepare data for the selected Model to be predicted
    def prepareDataForModel (self):

        # 1. Load the necessary Inputs

        # 1.1. Load the Adjacency Matrix
        loaded_adjMatrix = np.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\AdjacencyMatrix_train.npy")
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

        # 1.2. Load the scaler
        loaded_scaler = joblib.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\scaler.pkl")

        # 1.4. Load the last Layer
        loaded_last_layer = np.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\last_obs_layer.npy")
        print("DATA PREPARATION FOR PREDICTION - Shape of loaded Last Layer (for prediction shape): ", loaded_last_layer.shape)

        return 0

