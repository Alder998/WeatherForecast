# Class that allow to make prediction on future data and visualize them so to understand if the model has scored well
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from prophet import Prophet
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from DatabaseManager import Database as db
from DatabaseManager import DatabasePlugin_dask as dk
from DataPreparation import DataPreparation as dt
import tensorflow as tf
from pvlib import solarposition
from pvlib.solarposition import get_solarposition
import joblib

# Set silent option on downcasting to avoid warning
pd.set_option('future.no_silent_downcasting', True)

class PredictionService:

    def __init__(self, model, grid_step, start_date, prediction_steps, predictiveVariables, variableToPredict,
                 timeVariables):
        self.model = model
        self.grid_step = grid_step
        self.start_date = start_date
        self.prediction_steps = prediction_steps
        self.predictiveVariables = predictiveVariables
        self.variableToPredict = variableToPredict
        self.timeVariables = timeVariables
        pass

    # Utils-like function to standardize the data
    def standardizeData(self, set):

        # Initialize the scaler from scikit-learn
        scaler = StandardScaler()

        # Reshape that is necessary to process the features separately
        num_samples, num_obs, num_features = set.shape
        X_reshaped = set.reshape(-1, num_features)

        # Normalize the features
        X_scaled = scaler.fit_transform(X_reshaped)

        # Put the array in the old shape
        X_scaled = X_scaled.reshape(num_samples, num_obs, num_features)

        return X_scaled, scaler

    def dataClass (self, type='db'):

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
        else:
            raise Exception('The DB type ' + type + ' has been mispelled or it has not be implemented!')

        return dataClass

    # We want to make prediction ahead in time, therefore we must create a dataset in the same
    # shape as the model's one (using the space grid), and predict on that
    def createPredictionSet (self):

        # import the time grid
        gridPointData = self.dataClass().getDataFromTable("gridPoints_" + str(self.grid_step)).drop_duplicates()

        # Now, from the start date, create an hourly frame with the length of the prediction steps
        start_date = pd.to_datetime(self.start_date) + pd.Timedelta(days=1)
        predictionData = pd.DataFrame(pd.date_range(start=start_date, periods=self.prediction_steps,
                                                    freq='h')).set_axis(['date'], axis = 1)

        rawPredictionSet = []
        for gridPoint in range(len(gridPointData[gridPointData.columns[0]])):
            # Isolate the single couples of coordinates, concatenate them with the grid part, front-fill the NaN
            # Create the grid part isolating the couples of coordinates
            gridPart = gridPointData.iloc[gridPoint].reset_index().T
            gridPart = gridPart.set_axis([gridPart.loc['index']], axis = 1)[1:].reset_index(drop=True)
            gridPart.columns = gridPart.columns.get_level_values(0)
            # Now concatenate and front-fill
            dateWithGridCouple = pd.concat([predictionData, gridPart], axis = 1)
            dateWithGridCouple = dateWithGridCouple.ffill().infer_objects(copy=False)
            rawPredictionSet.append(dateWithGridCouple)
        rawPredictionSet = pd.concat([df for df in rawPredictionSet], axis = 0)

        # Change the latitude and longitude name in predictive variables list
        self.predictiveVariables = ["lat" if v == "latitude" else "lng" if v == "longitude" else v for v in self.predictiveVariables]

        # if required, add the solar angle
        if 'solar angle' in self.predictiveVariables:
            rawPredictionSet['key'] = rawPredictionSet['lat'].astype(str) + '_' + rawPredictionSet['lng'].astype(str)
            rawPredictionSet = dt.DataPreparation(self.grid_step).computeSolarInclinationFromDataFrame(rawPredictionSet)

        rawPredictionSet = rawPredictionSet.copy()
        rawPredictionSet['hour'] = pd.to_datetime(rawPredictionSet['date']).dt.hour
        rawPredictionSet['day'] = pd.to_datetime(rawPredictionSet['date']).dt.day
        rawPredictionSet['month'] = pd.to_datetime(rawPredictionSet['date']).dt.month

        # Add hour sin and cos
        if 'hour_sin' in self.predictiveVariables:
            print('computing hour sin...')
            rawPredictionSet = rawPredictionSet.copy()
            rawPredictionSet['hour_sin'] = np.sin(2 * np.pi * rawPredictionSet['hour'] / 24)
        if 'hour_cos' in self.predictiveVariables:
            print('computing hour cos...')
            rawPredictionSet = rawPredictionSet.copy()
            rawPredictionSet['hour_cos'] = np.cos(2 * np.pi * rawPredictionSet['hour'] / 24)
        if 'day_sin' in self.predictiveVariables:
            print('computing day sin...')
            rawPredictionSet = rawPredictionSet.copy()
            rawPredictionSet['day_sin'] = np.cos(2 * np.pi * rawPredictionSet['day'] / 24)
        if 'day_cos' in self.predictiveVariables:
            print('computing day cos...')
            rawPredictionSet = rawPredictionSet.copy()
            rawPredictionSet['day_cos'] = np.cos(2 * np.pi * rawPredictionSet['day'] / 24)
        if 'month_sin' in self.predictiveVariables:
            print('computing month sin...')
            rawPredictionSet = rawPredictionSet.copy()
            rawPredictionSet['month_sin'] = np.sin(2 * np.pi * rawPredictionSet['month'] / 24)
        if 'month_cos' in self.predictiveVariables:
            print('computing month cos...')
            rawPredictionSet = rawPredictionSet.copy()
            rawPredictionSet['month_cos'] = np.cos(2 * np.pi * rawPredictionSet['month'] / 24)
        if ('seasonal' in self.predictiveVariables) | ('trend' in self.predictiveVariables):
            # Create a Prophet prediction for each grid point
            prophetData = self.manageProphetPredictionForStagionality(prediction_set = rawPredictionSet,
                                                                       grid_points=len(gridPointData[gridPointData.columns[0]]))
            if 'seasonal' in self.predictiveVariables:
                rawPredictionSet = pd.concat([rawPredictionSet, prophetData["seasonal"]], axis = 1)
            if 'trend' in self.predictiveVariables:
                rawPredictionSet = pd.concat([rawPredictionSet, prophetData["trend"]], axis=1)

        # Delete the 'hour' column to avoid the double counting
        rawPredictionSet = rawPredictionSet.drop(columns = ['hour','day','month'])

        # Now, adapt the data for model with the appropriate function
        predictionSet = dt.DataPreparation(self.grid_step).adaptDataForModel(rawPredictionSet, self.predictiveVariables,
                                                                             labels=False, timeVariables=self.timeVariables)

        return predictionSet, rawPredictionSet

    # Class to load the model and run the pr]edictions
    def NNPredict (self, loaded_scaler=None, confidence_levels=False, n_iter=None):

        # Load the NN Model
        model = tf.keras.models.load_model(self.model)

        # Load the data creating a time-ahead set
        predictionSetRaw, predictionSet_df = self.createPredictionSet()
        # Adapt data for Model
        predictionSet = np.stack(predictionSetRaw, axis=0)
        predictedVariable = self.variableToPredict

        if confidence_levels==False:

            # Standardize data
            predictionSet, used_scaler = self.standardizeData(predictionSet)

            # Visualize the data size
            print('Prediction Set Size: ', self.getSetSize(predictionSet))

            # Make the Prediction
            predictions = model.predict(predictionSet)

            # Create the DataFrame for the prediction
            if loaded_scaler != None:
                targetScaler = joblib.load('scaler_labels_' + loaded_scaler + '.pkl')
            else:
                targetScaler = joblib.load('\\'.join(self.model.split('\\')[:-1]) + '\\scaler_labels_' + self.model.split('\\')[-1].replace('.h5','') + '.pkl')

            # Reshape to apply the inverse transform
            predictions_standardized_reshaped = predictions.reshape(-1, 1)  # (96 * 716, 1)
            predictions_original_scale = targetScaler.inverse_transform(predictions_standardized_reshaped)
            # Put the array in the original shape
            predictions_original_scale = predictions_original_scale.reshape(predictions.shape)

            # Convert to DataFrame
            # Find the variable name from model name to call the prediction column (Model name will have always the same order)
            df_prediction = pd.DataFrame(predictions_original_scale.reshape(-1, 1), columns=[predictedVariable])

            # If the variable predicted is the residuals, then add the prediction to the seasonal part
            if "_residual" in predictedVariable:
                df_prediction = (pd.Series(df_prediction[predictedVariable], index=predictionSet_df.index) + predictionSet_df["seasonal"])
                # then, change the column name for re-usability
                df_prediction = pd.DataFrame(df_prediction).rename(columns = {0 : predictedVariable})

            # Concatenate the prediction with the prediction set
            prediction_df_complete = pd.concat([predictionSet_df,
                                                df_prediction.dropna().set_index(predictionSet_df.index)], axis = 1)
            # Keep the useful columns, useful for the prediction display
            prediction_df_complete = prediction_df_complete[['date','lng','lat',predictedVariable]]
            prediction_df_complete = prediction_df_complete.set_axis(['date', 'longitude', 'latitude',
                                                                      predictedVariable], axis = 1)
            # Change the variable name to be shown in the report
            prediction_df_complete = prediction_df_complete.rename(columns = {predictedVariable : predictedVariable.replace("_residual", "")})

        else:
            n_iter = n_iter
            print('Running ' + str(n_iter) + ' Iterations with Montecarlo Dropout layer...')
            preds_mc = self.predict_with_uncertainty(model, predictionSet, n_iter=n_iter)

            # Computation of mean and std
            pred_mean = tf.reduce_mean(preds_mc, axis=0).numpy()  # shape: (batch, output)
            pred_std = tf.math.reduce_std(preds_mc, axis=0).numpy()  # shape: (batch, output)

            # Apply the inverse transform on mean and std
            # On the mean
            targetScaler = joblib.load('scaler_labels_' + self.model + '.pkl')
            pred_mean_reshaped = pred_mean.reshape(-1, 1)
            predictions_original_scale = targetScaler.inverse_transform(pred_mean_reshaped)
            predictions_original_scale = predictions_original_scale.reshape(pred_mean.shape)
            # On the standard deviation
            pred_std_reshaped = pred_std.reshape(-1, 1)
            predictions_std_original_scale = targetScaler.inverse_transform(pred_std_reshaped)
            predictions_std_original_scale = predictions_std_original_scale.reshape(pred_std.shape)

            # Confidence bands (95%)
            upper_bound = predictions_std_original_scale + 1.96 * predictions_std_original_scale
            lower_bound = predictions_std_original_scale - 1.96 * predictions_std_original_scale

            # DataFrame
            predictedVariable = self.model.split('_')[2]
            df_prediction = pd.DataFrame({
                predictedVariable: predictions_std_original_scale.reshape(-1),
                predictedVariable + '_std': predictions_std_original_scale.reshape(-1),
                predictedVariable + '_upper': upper_bound.reshape(-1),
                predictedVariable + '_lower': lower_bound.reshape(-1)
            })

            prediction_df_complete = pd.concat([predictionSet_df.reset_index(drop=True), df_prediction], axis=1)
            prediction_df_complete.columns = ['date', 'longitude', 'latitude',
                                              predictedVariable,
                                              predictedVariable + '_std',
                                              predictedVariable + '_upper',
                                              predictedVariable + '_lower']

        return prediction_df_complete

    # This function is to predict with confidence levels
    @tf.function
    def predict_with_uncertainty(self, f_model, x, n_iter=100):
        predictions = tf.stack([f_model(x, training=True) for _ in range(n_iter)], axis=0)
        return predictions  # Shape: (n_iter, batch, output)

    def getSetSize (self, set):

        tcheck = []
        for value in set:
            if value.shape not in tcheck:
                tcheck.append(value.shape)

        return tcheck

    def createProphetPrediction(self, prediction_set, dataset_depth=30, prediction_steps=1000):

        rootModelDirectory = "\\".join(self.model.split("\\")[:-1]) + "\\prophet_pred_" + self.variableToPredict.replace("_residual", "") + ".xlsx"
        # Case: the prophet prediction does not exist for the model: create it
        # get the data to train Prophet. To avoid it to be too long, take the last 10 days
        dataFromQuery = self.dataClass().executeQuery('SELECT * FROM public."WeatherForRegion_' + str(self.grid_step) +
                                             '" WHERE date BETWEEN ' + "'" + str(datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=dataset_depth)) + "'" + ' AND ' + "'" +
                                             self.start_date + "'")

        # Isolate each point into the grid
        dataWithPrediction = []
        dataFromQuery["key"] = dataFromQuery["latitude"].astype(str) + "_" + dataFromQuery["longitude"].astype(str)
        print("Adding Prophet Predictions to the model...")
        for i, singleCoord in enumerate(dataFromQuery["key"].unique()):
            dataFromQuery_time = dataFromQuery[dataFromQuery["key"] == singleCoord].reset_index(drop=True)
            # Sort by date
            dataFromQuery_time = dataFromQuery_time.sort_values(by="date").reset_index(drop=True)
            # Isolate the columns and rename it
            dataFromQuery_time = dataFromQuery_time[["date",
                                 self.variableToPredict.replace("_residual", "")]].rename(columns = {"date":"ds"}).rename(columns = {self.variableToPredict.replace("_residual", ""):"y"})
            # Fit prophet
            print("Adding Prophet Predictions to the model - " + str(round((i/len(dataFromQuery["key"].unique())) * 100, 2)) + "% ...")
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='additive',
                changepoint_prior_scale=1e-8
            )
            # fit the model
            m.fit(dataFromQuery_time)
            # Create future Dataframe + predict
            future = m.make_future_dataframe(periods=prediction_steps, freq='h')
            forecast = m.predict(future)
            forecast = forecast[forecast['ds'] > dataFromQuery_time['ds'].max()]

            # Extract Seasonality
            # Add trend to seasonality
            seasonality = (forecast['daily'] + forecast['weekly'] + forecast['yearly'] + forecast["trend"]).values
            # Extract Trend
            trend = pd.DataFrame(forecast["trend"].values).set_axis(["trend"], axis=1)
            seasonality = pd.DataFrame(seasonality).set_axis(["seasonal"], axis = 1)
            prophet_params = pd.concat([seasonality, trend], axis = 1)
            dataWithPrediction.append(prophet_params)
        dataWithPrediction = pd.concat([df for df in dataWithPrediction], axis=0).reset_index(drop=True)

        # Store the Prophet Prediction to avoid multiple computation for prediction (in the model directory)
        #rawPredictionSet = pd.concat([prediction_set, dataWithPrediction.set_index(prediction_set.index)], axis=1)
        dataWithPrediction.to_excel(rootModelDirectory)

        return dataWithPrediction

    def manageProphetPredictionForStagionality(self, prediction_set, grid_points):

        # Read the root model directory
        rootModelDirectory = "\\".join(self.model.split("\\")[:-1]) + "\\prophet_pred_" + self.variableToPredict.replace("_residual", "") + ".xlsx"

        # Case: the prophet prediction does not exist for the model
        if not os.path.exists(rootModelDirectory):
            predictionData = self.createProphetPrediction(prediction_set, dataset_depth=10, prediction_steps=100)
            return predictionData[:self.prediction_steps]
        # Case: the prophet prediction does exist for the model
        elif os.path.exists(rootModelDirectory):
            # read the file
            predictionData = pd.read_excel(rootModelDirectory)
            if len(predictionData[predictionData.columns[0]]) > (self.prediction_steps * grid_points):
                print("Reading and adapting existing xlsx-based prophet prediction...")
                return predictionData[:self.prediction_steps]
            else:
                # Update overwriting: no other choice
                predictionData = self.createProphetPrediction(prediction_set, dataset_depth=10, prediction_steps=100)
                return predictionData[:self.prediction_steps]