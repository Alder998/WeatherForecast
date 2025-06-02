# Data Preparation Class (mainly functional)
import math
import os
from datetime import datetime
from pvlib import solarposition
from pvlib.solarposition import get_solarposition
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from collections import Counter
import ModelService
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from geopy.distance import geodesic
from ModelStorageService import ModelStorageService as st

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

    # Utils-Like Function to compute the sun inclination given month, day, year, hour + coordinates
    def solar_inclination(self, row):
        time = pd.to_datetime(row['date'])
        lat = row['latitude']
        lng = row['longitude']
        solpos = get_solarposition(time, latitude=lat, longitude=lng)

        # different choices can be made
        return solpos['apparent_elevation'].values[0]

    # utils-like function to compute the solar angle in a faster way, in order to save computation and time
    def computeSolarInclinationFromDataFrame (self, dataframe):
        # For assumption, we can divide the dataFrame in 716 different single coordinates, that has been gathered in a
        # Column named "key"
        # Iterate for single key
        datasetWithSolarAngle = []
        for i, singleCoord in enumerate(dataframe['key'].unique()):
            print('Computing solar angle: ' + str(round((i / len(dataframe['key'].unique())) * 100, 2)) + '%')
            datasetFiltered = dataframe[dataframe['key'] == singleCoord].reset_index(drop=True)
            # df['lat'] + '_' + df['lng']
            datasetFiltered = datasetFiltered.copy()
            datasetFiltered['solar angle'] = solarposition.get_solarposition(datasetFiltered['date'],
                                             latitude=float(singleCoord.split('_')[0]),
                                             longitude=float(singleCoord.split('_')[1]))['apparent_elevation'].reset_index(drop = True)
            datasetWithSolarAngle.append(datasetFiltered)
        datasetWithSolarAngle = pd.concat([df for df in datasetWithSolarAngle], axis=0).reset_index(drop=True)
        # Fill the NaN with the ffill method (simply dropping them would be dangerous for model purposes)
        if len(datasetWithSolarAngle[datasetWithSolarAngle['solar angle'].isna()]) > 0:
            print('WARNING: Found: ' + str(len(datasetWithSolarAngle[datasetWithSolarAngle['solar angle'].isna()])) +
                  ' NaN Values while computing the solar angle - ffill wil be applied')
            datasetWithSolarAngle = datasetWithSolarAngle.copy()
            datasetWithSolarAngle['solar angle'] = datasetWithSolarAngle['solar angle'].ffill()
        return datasetWithSolarAngle

    def adaptDataForModel (self, dataFrame, predictiveVariables, labels, timeVariables=['year', 'month', 'day', 'hour']):

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

        # Now, compute the solar Angle
        if 'solar angle' in predictiveVariables:
            print('computing solar angle...')
            datasetFinal = self.computeSolarInclinationFromDataFrame(datasetFinal)

        if 'hour_sin' in predictiveVariables:
            print('computing hour sin...')
            datasetFinal = datasetFinal.copy()
            datasetFinal['hour_sin'] = np.sin(2 * np.pi * datasetFinal['hour'] / 24) * 0.5
        if 'hour_cos' in predictiveVariables:
            print('computing hour cos...')
            datasetFinal = datasetFinal.copy()
            datasetFinal['hour_cos'] = np.cos(2 * np.pi * datasetFinal['hour'] / 24) * 0.5
        if 'day_sin' in predictiveVariables:
            print('computing day sin...')
            datasetFinal = datasetFinal.copy()
            datasetFinal['day_sin'] = np.sin(2 * np.pi * datasetFinal['day'] / 24) * 0.5
        if 'day_cos' in predictiveVariables:
            print('computing day cos...')
            datasetFinal = datasetFinal.copy()
            datasetFinal['day_cos'] = np.cos(2 * np.pi * datasetFinal['day'] / 24) * 0.5
        if 'month_sin' in predictiveVariables:
            print('computing month sin...')
            datasetFinal = datasetFinal.copy()
            datasetFinal['month_sin'] = np.sin(2 * np.pi * datasetFinal['month'] / 24) * 0.5
        if 'month_cos' in predictiveVariables:
            print('computing month cos...')
            datasetFinal = datasetFinal.copy()
            datasetFinal['month_cos'] = np.cos(2 * np.pi * datasetFinal['month'] / 24) * 0.5

        # Isolate the columns of interest
        datasetFinal = datasetFinal[predictiveVariables]

        # Reshape the data
        print('Reshaping the data...')
        # Distinguish if labels or set
        if labels:
            # Target columns is just needed here
            targetColumn = datasetFinal.drop(columns=['year', 'month', 'day', 'hour']).columns[0]
            newSizeData = [group[targetColumn].to_numpy().reshape(-1, 1) for _,
            group in datasetFinal.groupby(['year', 'month', 'day', 'hour'])]
        else:
            newSizeData = [group.to_numpy() for _, group in datasetFinal.groupby(timeVariables)]

        return newSizeData

    # Utils-like method to get all same-shape arrays
    def cleanTrainAndTestSet (self, set):

        # Remove the data that are not in the desired shape
        # Extract the shapes
        shapes = [arr.shape for arr in set]
        # Count the shapes
        shape_counts = Counter(shapes)
        # find the most frequent shape
        most_common_shape = shape_counts.most_common(1)[0]
        # Now, filter
        newSet = [arr for arr in set if arr.shape == most_common_shape[0]]

        return newSet

    def getSpaceSplitNearPoints(self, gridPoints, test_size, nearPointsPerGroup = 4, plot_space_split = False):

        # For the geospatial purpose, we need to implement a special Train-Test setting
        # First, apply the Geo split (test: try to take 1 each 3 or 4 observations)

        # Space division: there is the need to find, set, model the test selecting NEAR POINTS
        if nearPointsPerGroup != 1:
            pointDivision = np.linspace(0, len(gridPoints[gridPoints.columns[0]]) - nearPointsPerGroup,
                                        int(len(gridPoints[gridPoints.columns[0]]) * test_size / nearPointsPerGroup))
        else:
            pointDivision = np.linspace(0, len(gridPoints[gridPoints.columns[0]]),
                                        int(len(gridPoints[gridPoints.columns[0]]) * test_size))

        # Make all the values inside the array integers
        pointDivision = [int(point) for point in pointDivision]

        if nearPointsPerGroup != 1:
            for addingPoint in range(1, nearPointsPerGroup + 1):
                pointDivision.extend([x + addingPoint for x in pointDivision])

        # TEMPORARY: "Block" the division size to the maximum available
        pointDivision = pointDivision[:int(len(gridPoints[gridPoints.columns[0]]) * (test_size))]

        # Filter for data Points inside and outside the Grid
        grid_train = gridPoints[~gridPoints.index.isin(pointDivision)].reset_index(drop=True)
        grid_test = gridPoints[gridPoints.index.isin(pointDivision)].reset_index(drop=True)

        if plot_space_split:
            plt.figure(figsize=(7, 7))
            plt.scatter(x=grid_train['lng'], y=grid_train['lat'], color='blue', label='Train Points')
            plt.scatter(x=grid_test['lng'], y=grid_test['lat'], color='red', label='Test Points')
            plt.title('Space Split on point grid')
            plt.legend()
            plt.show()

        return grid_train, grid_test

    def select_points_within_radius(self, df, center, radius_km):

        # Approximate Euclidean Distance
        df = df.copy()
        df['dist'] = np.sqrt((df['lng'] - center[0]) ** 2 + (df['lat'] - center[1]) ** 2)

        # Convert 111 km -> 1 rad
        km_per_degree = 111
        df['dist_km'] = df['dist'] * km_per_degree

        test_coord = df[df['dist_km'] <= radius_km].drop(columns=['dist', 'dist_km'])
        # Build the key
        df1 = df.copy()
        df1['key'] = df['lat'].astype(str) + '_' + df['lng'].astype(str)
        test_coord['key'] = test_coord['lat'].astype(str) + '_' + test_coord['lng'].astype(str)
        grid_test = df1[df1['key'].isin(test_coord['key'])].reset_index(drop=True)
        grid_train = df1[~df1['key'].isin(test_coord['key'])].reset_index(drop=True)

        # Take the necessary columns only
        grid_train = grid_train[['lat', 'lng']]
        grid_test = grid_test[['lat', 'lng']]

        return grid_train, grid_test

    def getSpaceSplitByRadius(self, gridPoints, test_size, radius_km, nearPointsPerGroup, plot_space_split=False):

        points = np.linspace(0, len(gridPoints[gridPoints.columns[0]]),
                             int((len(gridPoints[gridPoints.columns[0]]) * test_size) / nearPointsPerGroup))
        grid_train_r = []
        grid_test_r = []
        for p in points:
            p = int(p)
            if p < len(gridPoints[gridPoints.columns[0]]):
                tr_r, te_r = self.select_points_within_radius(df=gridPoints, center=gridPoints[p:p + 1].values[0],
                                                         radius_km=radius_km)
            grid_train_r.append(tr_r)
            grid_test_r.append(te_r)
        grid_train_r = pd.concat([df for df in grid_train_r], axis=0).drop_duplicates().reset_index(drop=True)
        grid_test_r = pd.concat([df for df in grid_test_r], axis=0).drop_duplicates().reset_index(drop=True)

        if plot_space_split:
            plt.figure(figsize=(7, 7))
            plt.scatter(x=grid_train_r['lng'], y=grid_train_r['lat'], color='blue', label='Train Points')
            plt.scatter(x=grid_test_r['lng'], y=grid_test_r['lat'], color='red', label='Test Points')
            plt.title('Space Split on point grid')
            plt.legend()
            plt.show()

        return grid_train_r, grid_test_r

    # Function for weekly time split not be affected by seasonality
    def timeSplit (self, dataset, test_size):

        # Create date range from start date to end date (returning n weeks sunday-sunday according the time range)
        weeklyRange = pd.date_range(pd.to_datetime(dataset['date']).min(), pd.to_datetime(dataset['date']).max(), freq='W')
        # Isolate the test set
        weeklyIndex = np.linspace(0, len(weeklyRange), int(len(weeklyRange) * test_size))
        weeklyRangeFiltered = []
        for index in weeklyIndex:
            if index < weeklyIndex.max():
                weeklyRangeFiltered.append(weeklyRange[int(index)])
            else:
                weeklyRangeFiltered.append(weeklyRange[int(index) - 1])

        # Create the week Adding 7 days (sundayW0-sundayW1)
        ranges = []
        if len(weeklyRangeFiltered) > 1:
            for wk in weeklyRangeFiltered:
                ranges.append([wk, wk + pd.Timedelta(days=7)])
        else:
            # Implement the test size with only one week of data
            ranges.append([weeklyRangeFiltered[0], weeklyRangeFiltered[0] + pd.Timedelta(days=7)])

        # Now, isolate the train and test set
        testSet = []
        for dateIndexCouple in ranges:
            # Isolated a couple of date indexes
            test_part = dataset[(pd.to_datetime(dataset['date']) >= dateIndexCouple[0]) & (pd.to_datetime(dataset['date']) <= dateIndexCouple[1])]
            testSet.append(test_part)
        testSet = pd.concat([df for df in testSet], axis = 0)#.reset_index(drop=True)

        # Generate the train set by exclusion
        trainSet = dataset[~dataset['date'].isin(testSet['date'])]#.reset_index(drop=True)

        # little bit of logging
        print('Time Split - Calculated % train size: ' +
              str(round(len(trainSet[trainSet.columns[0]]) / len(dataset[dataset.columns[0]]), 3) * 100) + '%')
        print('Time Split - Calculated % test size: ' +
              str(round(len(testSet[testSet.columns[0]]) / len(dataset[dataset.columns[0]]), 3) * 100) + '%')

        return trainSet, testSet

    def timeAndSpaceSplit (self, dataset, test_size, predictiveVariables, variableToPredict, space_split=True, time_split=True,
                           nearPointsPerGroup = 4, plot_space_split = False, space_split_method = 'uniform', timeVariables=['year', 'month', 'day', 'hour']):

        if (space_split == False) & (time_split == False):
            raise Exception ('Error! At least one split must be filled!')

        # Space Split
        # First, get the data for space split
        gridPoints = self.dataClass().getDataFromTable("gridPoints_" + str(self.grid_step)).drop_duplicates().reset_index(drop =True)
        if space_split_method == 'uniform':
            grid_train, grid_test = self.getSpaceSplitNearPoints(gridPoints = gridPoints, test_size=test_size,
                                                                 nearPointsPerGroup=nearPointsPerGroup,
                                                                 plot_space_split=plot_space_split)
        elif space_split_method == 'radius':
            grid_train, grid_test = self.getSpaceSplitByRadius(gridPoints = gridPoints, test_size=test_size,
                                                               radius_km=60, nearPointsPerGroup=nearPointsPerGroup,
                                                               plot_space_split = plot_space_split)
        else:
            raise Exception('Space Split Method not currently Implemented!')

        # Remove duplicates from original Dataset, and sort from the least recent to the most recent
        dataset = dataset.drop_duplicates(subset=['date', 'latitude', 'longitude']).reset_index(drop=True)
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset.sort_values(by='date', ascending=True).reset_index(drop=True)

        # Now apply the Time Split (train is the first n observations, test is the remaining m observations)
        if time_split:
            trainSet_time, testSet_time = self.timeSplit(dataset=dataset, test_size=test_size)
            if not space_split:
                trainTest, testSet = trainSet_time, testSet_time

        # Apply the train and test grid to the geo data, accordingly
        if space_split:
            # Create the key on general dataset
            dataset['key'] = dataset['latitude'].astype(str) + '_' + dataset['longitude'].astype(str)
            grid_train['key'] = grid_train['lat'].astype(str) + '_' + grid_train['lng'].astype(str)
            grid_test['key'] = grid_test['lat'].astype(str) + '_' + grid_test['lng'].astype(str)

            # exclude from train set the values present in test set (it may happen that some observation are common)
            grid_train = grid_train[~grid_train['key'].isin(grid_test['key'])].reset_index(drop=True)

            # Logging
            print('Space Split - Calculated % train size: ' +
                str(round(len(grid_train[grid_train.columns[0]]) / len(gridPoints[gridPoints.columns[0]]), 3) * 100) + '%')
            print('Space Split - Calculated % test size: ' +
                str(round(len(grid_test[grid_test.columns[0]]) / len(gridPoints[gridPoints.columns[0]]), 3) * 100) + '%')

            # If both train and test split flag have been activated, then this is the final split
            testSet_space = dataset[(dataset['key'].isin(grid_test['key']))]#.reset_index(drop=True)

            if not time_split:
                testSet = dataset[(dataset['key'].isin(grid_test['key']))].reset_index(drop=True)
                trainSet = dataset[(dataset['key'].isin(grid_train['key']))].reset_index(drop=True)

        if time_split and space_split:
            # General time-Space split - it is necessary to create two masks
            # Time mask
            mask_time = dataset['date'].isin(testSet_time['date'])
            # Space mask
            mask_space = (dataset['key'].isin(testSet_space['key']))

            mask_combined = mask_time | mask_space
            testSet = dataset[mask_combined]

            #testSet = pd.concat([testSet_time, testSet_space], axis = 0).drop_duplicates(subset = ['latitude','longitude','date'])#.reset_index(drop=True)
            #trainSet = dataset.loc[~dataset.index.isin(testSet.index)].reset_index(drop = True)
            # Filter the observations for the dates present in the train set and test set
            trainSet = dataset[~dataset.index.isin(testSet.index)]
            #testSet = testSet[testSet['date'].isin(testSet_time['date'])].reset_index(drop = True)

        # Logging
        print('Time-Space Split - Calculated % train size: ' +
           str(round((len(trainSet) - 2) / len(dataset[dataset.columns[0]]) * 100, 1)) + '%')
        print('Time-Space Split - Calculated % test size: ' +
           str(round((len(testSet) - 2) / len(dataset[dataset.columns[0]]) * 100, 1)) + '%')

        # Now, make the values as array
        train_set = self.adaptDataForModel(trainSet, predictiveVariables, labels = False, timeVariables=timeVariables)
        test_set = self.adaptDataForModel(testSet, predictiveVariables, labels = False, timeVariables=timeVariables)
        train_labels = self.adaptDataForModel(trainSet, ['year', 'month', 'day', 'hour', variableToPredict], labels = True, timeVariables=timeVariables)
        test_labels = self.adaptDataForModel(testSet, ['year', 'month', 'day', 'hour', variableToPredict], labels = True, timeVariables=timeVariables)

        # Remove the data that are not in the desired shape
        train_set = self.cleanTrainAndTestSet(train_set)
        train_labels = self.cleanTrainAndTestSet(train_labels)
        test_set = self.cleanTrainAndTestSet(test_set)
        test_labels = self.cleanTrainAndTestSet(test_labels)

        # Each one of the sets has a snapshot of the geographic area according to the time frame
        # REVISE: Make test and train set EVEN to make the Convolutional Model simpler and more flexible
        if (train_set[1].shape[0] % 2) != 0:
            train_set = [x[:-1, :] for x in train_set]
            train_labels = [x[:-1, :] for x in train_labels]
        if (test_set[1].shape[0] % 2) != 0:
            test_set = [x[:-1, :] for x in test_set]
            test_labels = [x[:-1, :] for x in test_labels]

        return train_set, test_set, train_labels, test_labels

    def getDataForModel (self, start_date, end_date, test_size, predictiveVariables, variableToPredict, modelName, space_split=True,
                         time_split=True, nearPointsPerGroup=4, plot_space_split=False, space_split_method='uniform',
                         timeVariables=['year', 'month', 'day', 'hour']):

        data = self.getDataWindow(start_date=start_date, end_date=end_date)
        # Train-test split
        train_set, test_set, train_labels, test_labels = self.timeAndSpaceSplit(dataset=data,
                                                                test_size=test_size,
                                                                predictiveVariables=predictiveVariables,
                                                                variableToPredict=variableToPredict,
                                                                space_split=space_split,
                                                                time_split=time_split,
                                                                nearPointsPerGroup = nearPointsPerGroup,
                                                                plot_space_split=plot_space_split,
                                                                space_split_method=space_split_method,
                                                                timeVariables=timeVariables)

        # Take care of saving the training data
        #Time - Split | Time - Space - Split | Space - Split
        if (time_split) & (space_split):
            split_method='Time-Space-Split'
        elif (time_split) & (not space_split):
            split_method='Time-Split'
        elif (space_split) & (not time_split):
            split_method='Space-Split'

        st.ModelStorageService(modelName=modelName).saveModelDataInfo(predictive_variables = predictiveVariables,
                                                                      split_method=split_method,
                                                                      geo_split=space_split_method,
                                                                      target_variable=variableToPredict,
                                                                      grid_step=self.grid_step,
                                                                      timeVariables=timeVariables)

        return train_set, test_set, train_labels, test_labels

    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck
