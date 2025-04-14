# Data Preparation Class (mainly functional)
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from geopy.distance import geodesic

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

    def adaptDataForModel (self, dataFrame, predictiveVariables, labels):

        # self.data is in DataFrame format, we need to transform it into array. The desired format is a three dimensional
        # Array, with two predictors (latitude, longitude) + the variable of interest, while the third dimension is given
        # by the time stamp. Imagine it like a t images of the area of interest, with the variable of interest,
        # where t is the time stamps

        # Read the data
        # Add to the predictive variables the variable to predict (target)
        # predictiveVariables.append(variableToPredict)

        # Take the three main date factors, that are year, month, day of month, hour of day
        datasetFinal = dataFrame.copy()
        datasetFinal['year'] = pd.to_datetime(datasetFinal['date']).dt.year
        datasetFinal['month'] = pd.to_datetime(datasetFinal['date']).dt.month
        datasetFinal['day'] = pd.to_datetime(datasetFinal['date']).dt.day
        datasetFinal['hour'] = pd.to_datetime(datasetFinal['date']).dt.hour
        # Isolate the columns of interest
        datasetFinal = datasetFinal[predictiveVariables]

        # Reshape the data
        print('Reshaping the data...')
        targetColumn = datasetFinal.drop(columns = ['year', 'month', 'day', 'hour']).columns[0]
        # Distinguish if labels or set
        if labels:
            newSizeData = [group[targetColumn].to_numpy().reshape(-1, 1) for _,
            group in datasetFinal.groupby(['year', 'month', 'day', 'hour'])]
        else:
            newSizeData = [group.to_numpy() for _, group in datasetFinal.groupby(['year', 'month', 'day', 'hour'])]

        return newSizeData

    def timeAndSpaceSplit (self, dataset, test_size, predictiveVariables, variableToPredict, space_split=True, time_split=True,
                           nearPointsPerGroup = 4):

        # check that space OR time split has been filled as True
        if (space_split == False) & (time_split == False):
            raise Exception ('Error! At least one split must be filled!')

        # For the geospatial purpose, we need to implement a special Train-Test setting
        # First, apply the Geo split (test: try to take 1 each 3 or 4 observations)
        # Take data from Time Grid (gridPoints_ + selected grid)
        gridPoints = self.dataClass().getDataFromTable("gridPoints_" + str(self.grid_step)).drop_duplicates().reset_index(drop =True)

        # Space division: there is the need to find, set, model the test selecting NEAR POINTS
        pointDivision = np.linspace(0, len(gridPoints[gridPoints.columns[0]]),
                                    int(len(gridPoints[gridPoints.columns[0]]) * test_size/nearPointsPerGroup))

        # Make all the values inside the array integers
        pointDivision = [int(point) for point in pointDivision]

        for addingPoint in range(1, nearPointsPerGroup + 1):
            pointDivision.extend([x + addingPoint for x in pointDivision])

        # TEMPORARY: "Block" the division size to the maximum available
        pointDivision = pointDivision[:int(len(gridPoints[gridPoints.columns[0]]) * (test_size))]

        # Filter for data Points inside and outside the Grid
        grid_train = gridPoints[~gridPoints.index.isin(pointDivision)].reset_index(drop=True)
        grid_test = gridPoints[gridPoints.index.isin(pointDivision)].reset_index(drop=True)

        # Remove duplicates from original Dataset, and sort from the least recent to the most recent
        dataset = dataset.drop_duplicates(subset=['date', 'latitude', 'longitude']).reset_index(drop=True)
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset.sort_values(by='date', ascending=True).reset_index(drop=True)

        # Now apply the Time Split (train is the first n observations, test is the remaining m observations)
        if time_split:
            trainSet = dataset[0:int(len(dataset) * (1-test_size))].reset_index(drop=True)
            testSet = dataset[int(len(dataset) * (1-test_size)):len(dataset)].reset_index(drop=True)

        # Apply the train and test grid to the geo data, accordingly
        if space_split:
            trainSet = trainSet[(trainSet['latitude'].isin(grid_train['lat'])) &
                                (trainSet['longitude'].isin(grid_train['lng']))]
            testSet = testSet[(testSet['latitude'].isin(grid_test['lat'])) &
                                (testSet['longitude'].isin(grid_test['lng']))]

        # Now, make the values as array
        train_set = self.adaptDataForModel(trainSet, predictiveVariables, labels = False)
        test_set = self.adaptDataForModel(testSet, predictiveVariables, labels = False)
        train_labels = self.adaptDataForModel(trainSet, ['year', 'month', 'day', 'hour', variableToPredict],
                                              labels = True)
        test_labels = self.adaptDataForModel(testSet, ['year', 'month', 'day', 'hour', variableToPredict],
                                             labels = True)

        # Each one of the sets has a snapshot of the geographic area according to the time frame
        # REVISE: Exclude the first and the last observation, to avoid to have different dimensions of data
        # REVISE 2: Make test and train set EVEN to make the Convolutional Model simpler and more flexible
        if (train_set[0].shape[0] % 2) != 0:
            train_set = [x[:-1, :] for x in train_set]
            train_labels = [x[:-1, :] for x in train_labels]
        if (test_set[0].shape[0] % 2) != 0:
            test_set = [x[:-1, :] for x in test_set]
            test_labels = [x[:-1, :] for x in test_labels]

        return train_set[1:-1], test_set[1:-1], train_labels[1:-1], test_labels[1:-1]

    def getDataForModel (self, start_date, end_date, test_size, predictiveVariables, variableToPredict, space_split=True,
                         time_split=True, nearPointsPerGroup=4):

        data = self.getDataWindow(start_date=start_date, end_date=end_date)
        # Train-test split
        train_set, test_set, train_labels, test_labels = self.timeAndSpaceSplit(dataset=data,
                                                                test_size=test_size,
                                                                predictiveVariables=predictiveVariables,
                                                                variableToPredict=variableToPredict,
                                                                space_split=space_split,
                                                                time_split=time_split,
                                                                nearPointsPerGroup = nearPointsPerGroup)
        return train_set, test_set, train_labels, test_labels

    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck
