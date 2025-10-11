# Class that allow to make prediction on future data and visualize them so to understand if the model has scored well
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from prophet import Prophet
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# Add all folders for batch execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from DataPreparation_graph import DataPreparation as dt
import tensorflow as tf
import joblib

# Set silent option on downcasting to avoid warning
pd.set_option('future.no_silent_downcasting', True)

class PredictionService:

    def __init__(self, ):
        pass

    # Main function to prepare data for the selected Model to be predicted
    def prepareDataForModel (self):

        # We are going to use the same classes on data preparation that we used in training phase, if possible
        dt.DataPreparation()

        return 0

