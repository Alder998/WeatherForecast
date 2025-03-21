# Data Preparation Class (mainly functional)
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk

class DataPreparation:

    def __init__(self, grid_step):
        self.grid_step = grid_step
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

    def getDataWindow (self, start_date, end_date):

        print('Getting the data...')
        # Here we are implementing a data getter framework to use filters directly in the query
        dataFromQuery = self.dataClass().executeQuery('SELECT * FROM public."WeatherForRegion_' + str(self.grid_step) +
                                             '" WHERE date BETWEEN ' + "'" + start_date + "'" + ' AND ' + "'" +
                                             end_date + "'")
        return dataFromQuery.drop(columns=['row_number'])

    def adaptDataForModel (self, dataFrame, predictiveVariables):

        # self.data is in DataFrame format, we need to transform it into array. The desired format is a three dimensional
        # Array, with two predictors (latitude, longitude) + the variable of interest, while the third dimension is given
        # by the time stamp. Imagine it like a t images of the area of interest, with the variable of interest,
        # where t is the time stamps

        # Read the data
        # Add to the predictive variables the variable to predict (target)
        # predictiveVariables.append(variableToPredict)
        # Now, take the columns of interest
        dataset = dataFrame[predictiveVariables]

        # Reshape the data
        print('Reshaping the data...')
        newSizeData = [dataset.to_numpy() for _, group in dataset.groupby('date')]

        return newSizeData

    def timeAndSpaceSplit (self, dataset, test_size, predictiveVariables, variableToPredict):

        # For the geospatial purpose, we need to implement a special Train-Test setting
        # First, apply the Geo split (test: try to take 1 each 3 or 4 observations)
        # Take data from Time Grid (gridPoints_0.22)
        gridPoints = self.dataClass().getDataFromTable("gridPoints_" + str(self.grid_step))

        pointDivision = np.linspace(0, len(gridPoints[gridPoints.columns[0]]),
                                    int(len(gridPoints[gridPoints.columns[0]]) * test_size))
        # Make all the values inside the array integers
        pointDivision = [int(point) for point in pointDivision]
        # Filter for data Points inside and outside the Grid
        grid_train = gridPoints[~gridPoints.index.isin(pointDivision)].reset_index(drop=True)
        grid_test = gridPoints[gridPoints.index.isin(pointDivision)].reset_index(drop=True)

        # Now apply the Time Split (train is the first n observations, test is the remaining m observations)
        trainSet = dataset[0:int(len(dataset) * (1-test_size))].reset_index(drop=True)
        testSet = dataset[int(len(dataset) * (1-test_size)):len(dataset)].reset_index(drop=True)

        # Apply the train and test grid to the geo data, accordingly
        trainSet = trainSet[(trainSet['latitude'].isin(grid_train['lat'])) &
                            (trainSet['longitude'].isin(grid_train['lng']))]
        testSet = testSet[(testSet['latitude'].isin(grid_test['lat'])) &
                            (testSet['longitude'].isin(grid_test['lng']))]

        # REVISE: make the data numeric
        trainSet['date'] = pd.Series(trainSet['date'].index).astype(int)
        testSet['date'] = pd.Series(testSet['date'].index).astype(int)

        # Now, make the values as array
        train_set = self.adaptDataForModel(trainSet, predictiveVariables)
        test_set = self.adaptDataForModel(testSet, predictiveVariables)
        train_labels = self.adaptDataForModel(trainSet, ['date', variableToPredict])
        test_labels = self.adaptDataForModel(testSet, ['date', variableToPredict])

        # Each one of the sets has a snapshot of the geographic area according to the time frame
        return train_set, test_set, train_labels, test_labels

    def getDataForModel (self, start_date, end_date, test_size, predictiveVariables, variableToPredict):

        data = self.getDataWindow(start_date=start_date, end_date=end_date)
        # Train-test split
        train_set, test_set, train_labels, test_labels = self.timeAndSpaceSplit(dataset=data,
                                                                test_size=test_size,
                                                                predictiveVariables=predictiveVariables,
                                                                variableToPredict=variableToPredict)
        return train_set, test_set, train_labels, test_labels
