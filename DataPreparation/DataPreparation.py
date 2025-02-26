# Data Preparation Class (mainly functional)
import os

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DatabaseManager import Database as db

class DataPreparation:

    def __init__(self, tableName, variableToPredict, predictiveVariables):
        self.tableName = tableName
        self.variableToPredict = variableToPredict
        self.predictiveVariables = predictiveVariables
        pass

    def databaseModule (self):

        env_path = r"D:\PythonProjects-Storage\WeatherForecast\App_core\app.env"
        load_dotenv(env_path)
        database = os.getenv("database")
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")

        # Instantiate the database Object
        dataClass = db.Database(database, user, password, host, port)

        return dataClass

    def adaptDataForModel (self, time_steps):

        # self.data is in DataFrame format, we need to transform it into array. The desired format is a three dimensional
        # Array, with two predictors (latitude, longitude) + the variable of interest, while the third dimension is given
        # by the time stamp. Imagine it like a t images of the area of interest, with the variable of interest,
        # where t is the time stamps

        # Read the data
        print('Reading the data...')
        dataset = self.databaseModule().getDataFromTable(self.tableName)
        dataset = dataset[self.predictiveVariables.append(self.variableToPredict)]

        print('Reshaping the data...')
        newSizeData = dataset.values.reshape(dataset[dataset.columns[0]], time_steps, len(dataset.columns[0]))

        return newSizeData

    def trainTestSplit (self):

        # For the geospatial purpose, we need to implement a special Train-Test setting

        return 0
