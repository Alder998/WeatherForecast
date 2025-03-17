# Data Preparation Class (mainly functional)
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk

class DataPreparation:

    def __init__(self):
        pass

    def dataClass (self, type ='db'):

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

    def getDataWindow (self, grid_step, start_date, end_date):

        print('Getting the data...')
        # Here we are implementing a data getter framework to use filters directly in the query
        dataFromQuery = self.dataClass().executeQuery('SELECT * FROM public."WeatherForRegion_' + str(grid_step) +
                                             '" WHERE date BETWEEN ' + "'" + start_date + "'" + ' AND ' + "'" +
                                             end_date + "'")
        return dataFromQuery.drop(columns=['row_number'])

    def adaptDataForModel (self, dataFrame, predictiveVariables, variableToPredict):

        # self.data is in DataFrame format, we need to transform it into array. The desired format is a three dimensional
        # Array, with two predictors (latitude, longitude) + the variable of interest, while the third dimension is given
        # by the time stamp. Imagine it like a t images of the area of interest, with the variable of interest,
        # where t is the time stamps

        # Read the data
        # Add to the predictive variables the variable to predict (target)
        predictiveVariables.append(variableToPredict)
        # Now, take the columns of interest
        dataset = dataFrame[predictiveVariables]

        # Reshape the data
        print('Reshaping the data...')
        newSizeData = [dataset.to_numpy() for _, group in dataset.groupby('date')]

        return newSizeData

    def timeAndSpaceSplit (self, arrayList):

        # For the geospatial purpose, we need to implement a special Train-Test setting
        # Time Split (train is the first n observations, test is the remaining m observations)


        return 0
