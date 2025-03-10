# Data Preparation Class (mainly functional)
import os

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk

class DataPreparation:

    def __init__(self, tableName, variableToPredict, predictiveVariables):
        self.tableName = tableName
        self.variableToPredict = variableToPredict
        self.predictiveVariables = predictiveVariables
        pass

    def databaseModule (self, type = 'db'):

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

    def adaptDataForModel (self, time_steps):

        # self.data is in DataFrame format, we need to transform it into array. The desired format is a three dimensional
        # Array, with two predictors (latitude, longitude) + the variable of interest, while the third dimension is given
        # by the time stamp. Imagine it like a t images of the area of interest, with the variable of interest,
        # where t is the time stamps

        # Read the data
        print('Reading the data...')
        dataset = self.databaseModule(type = 'dk').getDataFromTable(self.tableName)
        # Add to the predictive variables the variable to predict (target)
        self.predictiveVariables.append(self.variableToPredict)
        # Now, take the columns of interest
        dataset = dataset[self.predictiveVariables]

        print('Reshaping the data...')
        dataset_numpy = dataset.to_dask_array(lengths=True).compute()
        newSizeData = dataset_numpy.reshape(len(dataset[dataset.columns[0]]), time_steps, len(dataset.columns[0]))

        return newSizeData

    def trainTestSplit (self):

        # For the geospatial purpose, we need to implement a special Train-Test setting

        return 0
