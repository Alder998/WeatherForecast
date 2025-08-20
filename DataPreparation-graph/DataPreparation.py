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
import ModelService
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from geopy.distance import geodesic
from ModelStorageService import ModelStorageService as st

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

    # Utils-like function to get the train and test size
    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck

    # Utils-like function to process data from DataFrame to Graph
    def processDataFromDataFrameToGraph (self, dataInDataFrameFormat, variableToPredict=[], distance_threshold=100):

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

        # 2. Create the Feature Matrix: that is where you store your variables of interest
        for enumUniqueTS, uniqueTS in enumerate(dataInDataFrameFormat['date'].unique()):
            dataInDataFrameFormat.loc[dataInDataFrameFormat["date"] == uniqueTS, "time_index"] = enumUniqueTS
        # 2.1. Initialize the empty matrix for features: the shape must be (grid_steps, variables, time steps)
        feature_matrix = np.zeros((adj_matrix_norm.shape[0], len(variableToPredict), len(dataInDataFrameFormat['date'].unique())))

        # 2.2. Map indexes and nodes, and fill the matrix
        node_ids = {(lat, lon): i for i, (lat, lon) in enumerate(dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().values)}
        for idx, row in dataInDataFrameFormat.iterrows():
            i = node_ids[(row['latitude'], row['longitude'])]
            t = int(row["time_index"])
            for f_idx, feature in enumerate(variableToPredict):
                feature_matrix[i, f_idx, t] = row[feature]

        return adj_matrix_norm, feature_matrix

    # Main function to prepare data for graphs processing
    def prepareDataForGraphModel (self, start_date, end_date, variableToPredict):

        # 0. Get data from database
        print("DATA PREPARATION - Extracting data from Database...")
        dataInDataFrameFormat = self.getDataWindow(start_date=start_date, end_date=end_date)

        # 1. Transform from dataFrame to Graph
        print("DATA PREPARATION - Converting DataFrame into graph...")
        adj_matrix_norm, feature_matrix = self.processDataFromDataFrameToGraph(dataInDataFrameFormat=dataInDataFrameFormat,
                                                          distance_threshold=100,
                                                          variableToPredict=variableToPredict)
        print("DATA PREPARATION - INFO: Shape of normalized Adjacency Matrix: ", adj_matrix_norm.shape)
        print("DATA PREPARATION - INFO: Shape of Feature Matrix: ", feature_matrix.shape)

        # 2. time-space Split with appropriate libraries + obtain Model-ready params

        return adj_matrix_norm, feature_matrix
