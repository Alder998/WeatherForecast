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
from ModelService_graph.TensorFlowService import STBlock, DiffusionGraphConv, TemporalGatedBlock
import joblib


# Set silent option on downcasting to avoid warning
pd.set_option('future.no_silent_downcasting', True)

class PredictionService:

    def __init__(self, model):
        self.model = model
        pass

    # Main function to prepare data for the selected Model to be predicted
    def prepareDataForModel (self):

        # 1. Load the necessary Inputs
        # 1.1. Load the Model
        print("DATA PREPARATION FOR PREDICTION - Loading Model...")
        loaded_model = tf.keras.models.load_model("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\" + self.model,
                                                  custom_objects={"STBlock": STBlock.STBlock,
                                                                  "DiffusionGraphConv": DiffusionGraphConv.DiffusionGraphConv,
                                                                  "TemporalGatedBlock": TemporalGatedBlock.TemporalGatedBlock})

        # 1.2. Load the scaler
        loaded_scaler = joblib.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\scaler.pkl")

        # 1.3. Load the Adjacency Matrix
        loaded_adjMatrix = np.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\AdjacencyMatrix_train.npy")
        print("DATA PREPARATION FOR PREDICTION - Shape of loaded Adjacency Matrix: ", loaded_adjMatrix.shape)

        # 1.4. Load the last Layer
        loaded_last_layer = np.load("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + self.model + "\\last_obs_layer.npy")
        print("DATA PREPARATION FOR PREDICTION - Shape of loaded Last Layer (for prediction shape): ", loaded_last_layer.shape)

        return 0

