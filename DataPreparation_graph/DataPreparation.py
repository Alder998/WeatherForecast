# Data Preparation Class (mainly functional) adapted for Graphs-processing
import math
import os
import sys
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.tsa.seasonal import STL
from datetime import datetime
from pvlib import solarposition
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from geopy.distance import geodesic
from ModelStorageService import ModelStorageService as st
from sklearn.model_selection import GroupShuffleSplit

class DataPreparation:

    def __init__(self, grid_step):
        self.grid_step = grid_step
        pass

    # Utils function to handle SQL-Database
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

    # Utils-like function to extract data from Database given start and end date
    def getDataWindow (self, start_date, end_date):

        # Here we are implementing a data getter framework to use filters directly in the query
        dataFromQuery = self.dataClass().executeQuery('SELECT * FROM public."WeatherForRegion_' + str(self.grid_step) +
                                             '" WHERE date BETWEEN ' + "'" + start_date + "'" + ' AND ' + "'" +
                                             end_date + "'")
        return dataFromQuery.drop(columns=['row_number'])

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

    # Utils-like function to pad the feature matrix to the correct number of nodes
    def applyNodesPaddingForFeatures (self, feature_matrix, num_nodes_target):
        num_nodes, num_vars, window = feature_matrix.shape
        X_padded = np.zeros((num_nodes_target, num_vars, window), dtype=feature_matrix.dtype)
        X_padded[:num_nodes, :, :] = feature_matrix
        return X_padded

    # Utils-like function to pad the Adjacency matrix to the correct number of nodes
    def applyNodesPaddingForAdjacency (self, adj_matrix, num_nodes_target):
        num_nodes = adj_matrix.shape[0]
        A_padded = np.zeros((num_nodes_target, num_nodes_target), dtype=adj_matrix.dtype)
        A_padded[:num_nodes, :num_nodes] = adj_matrix
        return A_padded

    # Utils-like function to get the train and test size
    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck

    # Utils-like function to have a time-spacial split for the data
    def timeSpaceSplit (self, dataInDataFrameFormat, test_size=0.3, validation_size=0.15):

        # 1. Apply space split with nodes
        # 1.1. Create node_id
        dataInDataFrameFormat['node_id'] = dataInDataFrameFormat.groupby(['latitude', 'longitude']).ngroup()
        # 1.2. Split using sk-learn
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=1893)
        train_idx, test_idx = next(splitter.split(dataInDataFrameFormat, groups=dataInDataFrameFormat['node_id']))
        # 1.3. Divide the DataFrame
        df_train_spatial = dataInDataFrameFormat.iloc[train_idx]
        df_test_spatial = dataInDataFrameFormat.iloc[test_idx]

        # 2. Time split
        # 2.1. Sort by date
        df_train_spatial = df_train_spatial.sort_values('date')
        df_test_spatial = df_test_spatial.sort_values('date')

        # 2.2. Get the unique time stamps
        times = sorted(df_train_spatial['date'].unique())
        # 2.3 Create the thresholds for time
        train_end = int(len(times) * (1-test_size))
        val_end = int(len(times) * (1-test_size+validation_size))

        # 2.4. divide the DataFrame index
        train_time = times[:train_end]
        val_time = times[train_end:val_end]
        test_time = times[val_end:]

        # 2.5. Apply the time split
        train_final = df_train_spatial[df_train_spatial['date'].isin(train_time)]
        val_final = df_train_spatial[df_train_spatial['date'].isin(val_time)]
        test_final = df_test_spatial[df_test_spatial['date'].isin(test_time)]

        return train_final, test_final, val_final

    # Utils-like function to process data from DataFrame to Graph
    def createAdjacencyMatrix (self, dataInDataFrameFormat, padding_target, distance_threshold=100):

        # 1. Create the Adjacency matrix
        # 1.1. Compute node id as incremental index
        nodeMapping = dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
        nodeMapping = nodeMapping.reset_index()
        nodeMapping = nodeMapping.rename(columns={"index" : "node_id"})

        # 1.2. Compute the distance matrix in km
        coords = np.radians(nodeMapping[['latitude', 'longitude']].values)
        dist_matrix = haversine_distances(coords, coords) * 6371.0

        # 1.3. Compute the weighted Adjacency Matrix
        adj_matrix = (dist_matrix <= distance_threshold).astype(float)
        sigma = dist_matrix[dist_matrix < np.inf].std()
        # Apply Gaussian Kernel for gradual weights
        adj_matrix = np.exp(- dist_matrix ** 2 / (2 * sigma ** 2)) * adj_matrix
        np.fill_diagonal(adj_matrix, 0)

        # 1.4. Normalize Matrix
        D_hat = np.diag(np.sum(adj_matrix + np.eye(adj_matrix.shape[0]), axis=1))
        D_inv = np.linalg.inv(D_hat)
        adj_matrix_norm = D_inv @ (adj_matrix + np.eye(adj_matrix.shape[0]))

        # Apply padding to achieve the same size
        adj_matrix_norm = self.applyNodesPaddingForAdjacency(adj_matrix_norm, padding_target)

        return adj_matrix_norm

    def createFeaturesMatrix (self, dataInDataFrameFormat, padding_target, variableToPredict=[]):

        # 1. get the number of unique coords
        coords = len(dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().values)

        # 2. Create the Feature Matrix: that is where you store your variables of interest
        for enumUniqueTS, uniqueTS in enumerate(dataInDataFrameFormat['date'].unique()):
            dataInDataFrameFormat.loc[dataInDataFrameFormat["date"] == uniqueTS, "time_index"] = enumUniqueTS
        # 2.1. Initialize the empty matrix for features: the shape must be (grid_steps, variables, time steps)
        feature_matrix = np.zeros((coords, len(variableToPredict), len(dataInDataFrameFormat['date'].unique())))

        # 2.2. Map indexes and nodes, and fill the matrix
        node_ids = {(lat, lon): i for i, (lat, lon) in enumerate(dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().values)}
        for idx, row in dataInDataFrameFormat.iterrows():
            i = node_ids[(row['latitude'], row['longitude'])]
            t = int(row["time_index"])
            for f_idx, feature in enumerate(variableToPredict):
                feature_matrix[i, f_idx, t] = row[feature]

        # Apply Padding for features
        feature_matrix = self.applyNodesPaddingForFeatures(feature_matrix=feature_matrix, num_nodes_target=padding_target)

        return feature_matrix

    # Function to create model-ready tensors
    def createModelTensors (self, set, window_size, horizon):

        # 1. Initialize the shapes as you have them
        num_nodes, num_features, time_steps = set.shape
        num_samples = time_steps - window_size - horizon + 1

        # 2. Initialize the tensors to be used as set and targets
        sample = np.zeros((num_samples, num_nodes, num_features, window_size))
        target = np.zeros((num_samples, num_nodes, num_features, horizon))

        # 3. create the rolling windows
        for i in range(num_samples):
            sample[i] = set[:, :, i: i + window_size]
            target[i] = set[:, :, i + window_size: i + window_size + horizon]

        return sample, target

    # Main function to prepare data for graphs processing
    def prepareDataForGraphModel (self, start_date, end_date, variableToPredict, test_size, validation_size,
                                  window_size, horizon, distance_threshold):

        # 0. Get data from database
        print("DATA PREPARATION - Extracting data from Database...")
        dataInDataFrameFormat = self.getDataWindow(start_date=start_date, end_date=end_date)
        # 0.1. Extract the total number of coordinates to use it during padding
        paddingTargetNodes = len(dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().values)

        # 1. time-space Split with appropriate libraries
        train_set, test_set, validation_set = self.timeSpaceSplit(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                 test_size=test_size,
                                                                 validation_size=validation_size)

        # 2. Create Adjacency Matrix for each one of the sets (the dimensions are padded)
        print("DATA PREPARATION - Converting DataFrame into graph...")
        adj_matrix_norm_train = self.createAdjacencyMatrix(dataInDataFrameFormat=train_set, distance_threshold=distance_threshold, padding_target=paddingTargetNodes)
        print("DATA PREPARATION - INFO (TRAIN SET): Shape of normalized Adjacency Matrix: ", adj_matrix_norm_train.shape)
        adj_matrix_norm_test = self.createAdjacencyMatrix(dataInDataFrameFormat=test_set, distance_threshold=distance_threshold, padding_target=paddingTargetNodes)
        print("DATA PREPARATION - INFO (TEST SET): Shape of normalized Adjacency Matrix: ", adj_matrix_norm_test.shape)
        adj_matrix_norm_validation = self.createAdjacencyMatrix(dataInDataFrameFormat=validation_set, distance_threshold=distance_threshold, padding_target=paddingTargetNodes)
        print("DATA PREPARATION - INFO (VALIDATION SET): Shape of normalized Adjacency Matrix: ", adj_matrix_norm_validation.shape)

        # 3. Create feature Matrix for each one of the sets
        feature_matrix_train = self.createFeaturesMatrix(dataInDataFrameFormat=train_set,
                                                         variableToPredict=variableToPredict,
                                                         padding_target=paddingTargetNodes)
        feature_matrix_test = self.createFeaturesMatrix(dataInDataFrameFormat=test_set,
                                                        variableToPredict=variableToPredict,
                                                        padding_target=paddingTargetNodes)
        feature_matrix_validation = self.createFeaturesMatrix(dataInDataFrameFormat=validation_set,
                                                              variableToPredict=variableToPredict,
                                                              padding_target=paddingTargetNodes)

        # 4. Create model-ready tensors
        sample_train, target_train = self.createModelTensors(set=feature_matrix_train, window_size=window_size, horizon=horizon)
        print("DATA PREPARATION - INFO (TRAIN SET): Shape of Sample Matrix: ", sample_train.shape)
        print("DATA PREPARATION - INFO (TRAIN SET): Shape of Target Matrix: ", target_train.shape)
        sample_test, target_test = self.createModelTensors(set=feature_matrix_test, window_size=window_size, horizon=horizon)
        print("DATA PREPARATION - INFO (TEST SET): Shape of Sample Matrix: ", sample_test.shape)
        print("DATA PREPARATION - INFO (TEST SET): Shape of Target Matrix: ", target_test.shape)
        sample_validation, target_validation = self.createModelTensors(set=feature_matrix_validation, window_size=window_size, horizon=horizon)
        print("DATA PREPARATION - INFO (VALIDATION SET): Shape of Sample Matrix: ", sample_validation.shape)
        print("DATA PREPARATION - INFO (VALIDATION SET): Shape of Target Matrix: ", target_validation.shape)

        return adj_matrix_norm_train, adj_matrix_norm_test, adj_matrix_norm_validation, sample_train, target_train, sample_test, target_test, sample_validation, target_validation
