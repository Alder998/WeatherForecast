# Data Preparation Class (mainly functional) adapted for Graphs-processing

import os
import joblib
from sklearn.metrics.pairwise import haversine_distances
from pvlib import solarposition
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from UserService import User_getter as user

class DataPreparation:

    def __init__(self, grid_step, environment):
        self.grid_step = grid_step
        self.environment = environment
        pass

    # Utils-like function to standardize the data and save the scaler
    def standardizeSet (self, set, axis=3, save_name="scaler"):

        # Set the axis to standardize the features
        matrixDimension = set.shape[axis]  # = 2

        # Put the feature at the end to be standardized
        X_reshaped = set.transpose(0, 1, 3, 2).reshape(-1, matrixDimension)

        # Fit scaler on F
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)

        # Put everything into the original form
        X_scaled = X_scaled.reshape(set.shape[0], set.shape[1], set.shape[3], set.shape[2]).transpose(0, 1, 3, 2)

        # Save the scaler + return the scaled numpy object
        if save_name != "None":
            if self.environment == "local":
                joblib.dump(scaler, user.user_getter(user=self.environment) + "Stored-models\\" + save_name + "\\scaler.pkl")
            elif self.environment == "cola-drive":
                joblib.dump(scaler, user.user_getter(user=self.environment) + "Stored-models/" + save_name + "/scaler.pkl")
            else:
                raise Exception("The environment: " + str() + " does not exist! available 'local' | 'colab-drive'")

            print("INFO - Scaler Saved successfully within the model class.")

        return X_scaled

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

    def getDataWindowFromCSV (self, start_date, end_date):

        # Here we are implementing a .csv reader
        print("MODEL DATA - Reading from .csv stored data...")
        csv_data = pd.read_csv(user.user_getter(self.environment) + "weatherForecast.csv")
        csv_data.columns = ["date", "latitude", "longitude", "temperature", "precipitation", "humidity_mean",
                            "windSpeed", "cloudCover", "pressure_msl", "row_number"]
        # Filter for start and end dates
        print("Filtering .csv data for selected dates...")
        data_filtered = csv_data[pd.to_datetime(csv_data["date"]).between(start_date, end_date)]
        data_filtered = data_filtered.sort_values(by="date", ascending=True).reset_index(drop=True)
        return data_filtered.drop(columns=['row_number'])

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

    # Utils-like function to get the train and test size
    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck

    # Utils-like function to apply time-split (no space split)
    def timeSplit(self, dataInDataFrameFormat, test_size=0.3, validation_size=0.15):

        # 1. Apply space split with nodes
        # 1.1. Create node_id
        dataInDataFrameFormat['node_id'] = dataInDataFrameFormat.groupby(['latitude', 'longitude']).ngroup()

        # 2.2. Get the unique time stamps
        times = sorted(dataInDataFrameFormat['date'].unique())
        # 2.3 Create the thresholds for time
        train_end = int(len(times) * (1-test_size))
        val_end = int(len(times) * (1-test_size+validation_size))

        # 2.4. divide the DataFrame index
        train_time = times[:train_end]
        val_time = times[train_end:val_end]
        test_time = times[val_end:]

        # 2.5. Apply the time split
        train_final = dataInDataFrameFormat[dataInDataFrameFormat['date'].isin(train_time)]
        val_final = dataInDataFrameFormat[dataInDataFrameFormat['date'].isin(val_time)]
        test_final = dataInDataFrameFormat[dataInDataFrameFormat['date'].isin(test_time)]

        return train_final, test_final, val_final

    # Utils-like function to process data from DataFrame to Graph
    def createAdjacencyMatrix (self, dataInDataFrameFormat, save_name, distance_threshold=100):

        # 1. Create the Adjacency matrix
        # 1.1. Compute node id as incremental index
        nodeMapping = dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True).sort_values(['latitude', 'longitude'])
        # Save the coords order
        if self.environment == "local":
            np.save(user.user_getter(user=self.environment) + "Stored-models\\" + save_name + "\\coords_order.npy",nodeMapping.values)
        elif self.environment == "colab-drive":
            np.save(user.user_getter(user=self.environment) + "Stored-models/" + save_name + "/coords_order.npy",nodeMapping.values)
        else:
            raise Exception("The environment: " + str() + " does not exist! available 'local' | 'colab-drive'")

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

        # Apply self-loop
        adj_matrix_norm = adj_matrix_norm.copy()
        adj_matrix_norm = adj_matrix_norm + np.eye(adj_matrix_norm.shape[0])
        # Insert significant connection if 0
        thr = 1e-5
        A_thr = np.where(adj_matrix_norm > thr, adj_matrix_norm, 0.0)
        # Normalization
        d = A_thr.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        adj_matrix_norm = (d_inv_sqrt[:, None] * A_thr) * d_inv_sqrt[None, :]

        # Memorize the set size in a dict together with the matrix
        adj_matrix_norm_data = {}
        adj_matrix_norm_data["size"] = adj_matrix_norm.shape[0]  # could be both 0 or 1, since the matrix is squared

        # Apply padding to achieve the same size
        # as well, store the matrix into the dict
        adj_matrix_norm_data["matrix"] = adj_matrix_norm

        # Check for disconnected points within the adjusted Adjacency Matrix
        G = nx.from_numpy_array(adj_matrix_norm)
        comps = list(nx.connected_components(G))
        print("INFO - CHECK ON ADJUSTED MATRIX: Connected Points:", len(comps), "(No issues)" if len(comps) == 1 else "(Warning)")

        return adj_matrix_norm_data

    # Utils-like function to create KNN-based Adjacency Matrix
    def createKNNAdjacencyMatrix (self, dataInDataFrameFormat, save_name, nn=10):

        # 1. Create the Adjacency matrix
        # 1.1. Compute node id as incremental index
        nodeMapping = dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True).sort_values(['latitude', 'longitude'])
        # Save the coords order
        if self.environment == "local":
            np.save(user.user_getter(user=self.environment) + "Stored-models\\" + save_name + "\\coords_order.npy", nodeMapping.values)
        elif self.environment == "colab-drive":
            np.save(user.user_getter(user=self.environment) + "Stored-models/" + save_name + "/coords_order.npy",nodeMapping.values)
        else:
            raise Exception("The environment: " + str() + " does not exist! available 'local' | 'colab-drive'")

        # N must be the size of the square matrix (if grid=0.22 it must be 716)
        N = len(nodeMapping[nodeMapping.columns[0]])

        nbrs = NearestNeighbors(n_neighbors=nn + 1).fit(nodeMapping)
        distances, indices = nbrs.kneighbors(nodeMapping)
        # Build an Adjacency Matrix
        A_knn = np.zeros((N, N))
        for i in range(N):
            for j in indices[i, 1:]:  # skip self
                A_knn[i, j] = 1
        # Make it simmetric
        A_knn = np.maximum(A_knn, A_knn.T)

        # Memorize the set size in a dict together with the matrix
        adj_matrix_norm_data = {}
        adj_matrix_norm_data["size"] = A_knn.shape[0]  # could be both 0 or 1, since the matrix is squared

        # Add self loop
        A_knn = A_knn + np.eye(A_knn.shape[0])

        # as well, store the matrix into the dict
        adj_matrix_norm_data["matrix"] = A_knn

        # Check for disconnected nodes
        row_sums = A_knn.sum(axis=1)
        low_deg_nodes = np.where(row_sums < np.percentile(row_sums, 5))[0]
        print("INFO - CHECK ON ADJUSTED MATRIX - Low-degree nodes:", len(low_deg_nodes), "(No issues)" if len(low_deg_nodes) == 0 else "(Warning)")

        # Check for disconnected points within the adjusted Adjacency Matrix
        G = nx.from_numpy_array(A_knn)
        comps = list(nx.connected_components(G))
        print("INFO - CHECK ON ADJUSTED MATRIX - Connected Points:", len(comps), "(No issues)" if len(comps) == 1 else "(Warning)")

        return adj_matrix_norm_data

    def createFeaturesMatrix (self, dataInDataFrameFormat, save_name, variableToPredict=[]):

        # 0. Load the coords order to avoid mismatches with different coords
        if self.environment=="local":
            coords_ref = np.load( user.user_getter(user=self.environment) + "Stored-models\\" + save_name + "\\coords_order.npy")
        elif self.environment == "colab-drive":
            coords_ref = np.load( user.user_getter(user=self.environment) + "Stored-models/" + save_name + "/coords_order.npy")
        else:
            raise Exception("The environment: " + str() + " does not exist! available 'local' | 'colab-drive'")

        # 1. get the number of unique coords
        coords = len(dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().values)

        # 2. Create the Feature Matrix: that is where you store your variables of interest
        dataInDataFrameFormat1 = dataInDataFrameFormat.copy()
        for enumUniqueTS, uniqueTS in enumerate(dataInDataFrameFormat1['date'].unique()):
            dataInDataFrameFormat1.loc[dataInDataFrameFormat1["date"] == uniqueTS, "time_index"] = enumUniqueTS
        # 2.1. Initialize the empty matrix for features: the shape must be (grid_steps, variables, time steps)
        feature_matrix = np.zeros((len(coords_ref), len(variableToPredict), len(dataInDataFrameFormat1['date'].unique())))

        # 2.2. Map indexes and nodes, and fill the matrix
        node_ids = {tuple(c): i for i, c in enumerate(coords_ref)}
        for idx, row in dataInDataFrameFormat1.iterrows():
            key = (row['latitude'], row['longitude'])
            if key not in node_ids:
                continue
            i = node_ids[key]
            t = int(row["time_index"])
            for f_idx, feature in enumerate(variableToPredict):
                feature_matrix[i, f_idx, t] = row[feature]

        # As done with the adjacency matrix, store the size and the matrix in a dict
        feature_matrix_data = {}
        feature_matrix_data["size"] = feature_matrix.shape[0]

        # Apply Padding for features
        #feature_matrix = self.applyNodesPaddingForFeatures(feature_matrix=feature_matrix, num_nodes_target=padding_target)
        # Now store the matrix itself
        feature_matrix_data["matrix"] = feature_matrix

        return feature_matrix_data

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
                                  window_size, horizon, save_name, matrix_params={}, all_data_scaler=False):

        # First, create model directory, if it does not exist
        if not os.path.exists(user.user_getter(user=self.environment) + "Stored-models\\" + save_name):
            if self.environment == "local":
                os.mkdir(user.user_getter(user=self.environment) + "Stored-models\\" + save_name)
            elif self.environment == "colab-drive":
                os.mkdir(user.user_getter(user=self.environment) + "Stored-models/" + save_name)
            else:
                raise Exception("The environment: " + str() + " does not exist! available 'local' | 'colab-drive'")

        # 0. Get data from database
        print("DATA PREPARATION - Extracting data from Database...")
        if self.environment == "local":
            dataInDataFrameFormat = self.getDataWindow(start_date=start_date, end_date=end_date).dropna()
        elif self.environment == "colab-drive":
            dataInDataFrameFormat = self.getDataWindowFromCSV(start_date=start_date, end_date=end_date).dropna()
        else:
            raise Exception("User " + str(user) + " not implemented!")
        # 0.1. Extract the total number of coordinates to use it during padding
        paddingTargetNodes = len(dataInDataFrameFormat[['latitude', 'longitude']].drop_duplicates().values)

        # 1. time-space Split with appropriate libraries
        train_set, test_set, validation_set = self.timeSplit(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                 test_size=test_size,
                                                                 validation_size=validation_size)

        # 2. Create Adjacency Matrix for each one of the sets (the dimensions are padded)
        print("DATA PREPARATION - Converting DataFrame into graph...")
        if matrix_params["type"] == "distance":
            adj_matrix_norm = self.createAdjacencyMatrix(dataInDataFrameFormat=dataInDataFrameFormat, distance_threshold=matrix_params["threshold"], save_name=save_name)
        elif matrix_params["type"] == "KNN":
            adj_matrix_norm = self.createKNNAdjacencyMatrix(dataInDataFrameFormat=dataInDataFrameFormat, nn=matrix_params["threshold"], save_name=save_name)
        else:
            raise Exception("ERROR! the calculation type: " + matrix_params["type"] + " has not been implemented!")

        print("DATA PREPARATION - INFO: Shape of normalized (unique) Adjacency Matrix: ", adj_matrix_norm["matrix"].shape, " - Non-zero points: ", adj_matrix_norm["size"])

        # 2.1. Save the adjacency matrix used for training in .npy format
        if self.environment == "local":
            np.save(user.user_getter(user=self.environment) + "Stored-models\\" + save_name + "\\AdjacencyMatrix.npy", adj_matrix_norm["matrix"])
        elif self.environment == "colab-drive":
            np.save(user.user_getter(user=self.environment) + "Stored-models/" + save_name + "/AdjacencyMatrix.npy", adj_matrix_norm["matrix"])
        else:
            raise Exception("The environment: " + str() + " does not exist! available 'local' | 'colab-drive'")

        # 3. Create feature Matrix for each one of the sets
        feature_matrix_train = self.createFeaturesMatrix(dataInDataFrameFormat=train_set,
                                                         variableToPredict=variableToPredict,
                                                         save_name=save_name)
        feature_matrix_test = self.createFeaturesMatrix(dataInDataFrameFormat=test_set,
                                                        variableToPredict=variableToPredict,
                                                        save_name=save_name)
        feature_matrix_validation = self.createFeaturesMatrix(dataInDataFrameFormat=validation_set,
                                                              variableToPredict=variableToPredict,
                                                              save_name=save_name)
        # Save the Scaler with the modelService
        if all_data_scaler:
            print("DATA PREPARATION - saving the all-data scaler...")
            feature_matrix_all = self.createFeaturesMatrix(dataInDataFrameFormat=dataInDataFrameFormat,
                                                             variableToPredict=variableToPredict,
                                                             save_name=save_name)
            sample_all, target_all = self.createModelTensors(set=feature_matrix_all["matrix"], window_size=window_size, horizon=horizon)
            self.standardizeSet(sample_all, axis=2, save_name=save_name)

        # 4. Create model-ready tensors
        sample_train, target_train = self.createModelTensors(set=feature_matrix_train["matrix"], window_size=window_size, horizon=horizon)
        print("DATA PREPARATION - INFO (TRAIN SET): Shape of Sample Matrix: ", sample_train.shape, "- steps without padding: ", feature_matrix_train["size"])
        print("DATA PREPARATION - INFO (TRAIN SET): Shape of Target Matrix: ", target_train.shape)
        sample_test, target_test = self.createModelTensors(set=feature_matrix_test["matrix"], window_size=window_size, horizon=horizon)
        print("DATA PREPARATION - INFO (TEST SET): Shape of Sample Matrix: ", sample_test.shape, "- steps without padding: ", feature_matrix_test["size"])
        print("DATA PREPARATION - INFO (TEST SET): Shape of Target Matrix: ", target_test.shape)
        sample_validation, target_validation = self.createModelTensors(set=feature_matrix_validation["matrix"], window_size=window_size, horizon=horizon)
        print("DATA PREPARATION - INFO (VALIDATION SET): Shape of Sample Matrix: ", sample_validation.shape, "- steps without padding: ", feature_matrix_validation["size"])
        print("DATA PREPARATION - INFO (VALIDATION SET): Shape of Target Matrix: ", target_validation.shape)

        return adj_matrix_norm, sample_train, target_train, sample_test, target_test, sample_validation, target_validation
