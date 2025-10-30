# This class is to create a Neural Network Library Built upon TensorFlow

import numpy as np
from keras.src.layers import Input
from tensorflow.keras import layers, models
import tensorflow as tf
from TensorFlowService import STBlock as stb
from TensorFlowService import GraphWaveNet as gwn
from ModelStorageService import ModelStorageService as st
import json
import os
import joblib
import DataPreparation_graph as dt
from sklearn.preprocessing import StandardScaler


class ModelService:

    def __init__(self, train_set, test_set, train_labels, test_labels, validation_set, validation_labels):
        self.train_set = train_set
        self.test_set = test_set
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.validation_set = validation_set
        self.validation_labels = validation_labels
        pass

    # Utils-like function to standardize the sets according to a given dimensions
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
        joblib.dump(scaler, "D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name + "\\scaler.pkl")

        return X_scaled

    # Utils-like function to create a custom loss that would ignore padding during training phase
    def masked_mse(self, y_true, y_pred, mask):
        mask = tf.reshape(mask, (1, -1, 1, 1))
        diff = (y_true - y_pred) * mask
        numerator = tf.reduce_sum(tf.square(diff))
        denominator = tf.reduce_sum(mask) * tf.cast(tf.shape(y_true)[0], tf.float32) * tf.cast(tf.shape(y_true)[-1],tf.float32)
        return numerator / (denominator + 1e-8)

    # Utils-like function to create node mask to ignore padding during training phase
    def create_node_mask(self, num_nodes_valid, num_nodes_target):
        mask = np.zeros((num_nodes_target,), dtype=np.float32)
        mask[:num_nodes_valid] = 1.0
        return mask

    def WaveNetTimeSpaceModel (self, adj_matrix, model_params, training_epochs, variableToPredict,
                               end_date, save_name="model"):

        # First, create model directory, if it does not exist
        if not os.path.exists("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name):
            os.mkdir("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name)

        # Extract Dimensions from input set
        N_train = self.train_set.shape[1]
        F_in = self.train_set.shape[2]  # ex. 1
        W = self.train_set.shape[3]  # ex. 24
        H = self.train_labels.shape[3]  # ex. 96

        # Standardize each one of the sets
        print("MODEL PREPARATION - Standardizing the sets...")
        self.train_set = self.standardizeSet(self.train_set, axis=2, save_name=save_name)
        self.train_labels = self.standardizeSet(self.train_labels, axis=2, save_name=save_name)
        self.test_set = self.standardizeSet(self.test_set, axis=2, save_name=save_name)
        self.test_labels = self.standardizeSet(self.test_labels, axis=2, save_name=save_name)
        self.validation_set = self.standardizeSet(self.validation_set, axis=2, save_name=save_name)
        self.validation_labels = self.standardizeSet(self.validation_labels, axis=2, save_name=save_name)

        # n_blocks has to be the same as the length of the dilations tuple (for b, d in enumerate(dilations[:n_blocks]))
        n_blocks = len(model_params["dilations"])

        model = gwn.GraphWaveNet(N=N_train, F_in=F_in, W=W, H=H, A=adj_matrix["matrix"],
                        channels_t=model_params["channels_t"], channels_s=model_params["channels_s"],
                        n_blocks=n_blocks, dilations=model_params["dilations"],
                        kernel_size=model_params["kernel_size"]).build_graph_wavenet()

        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.metrics.MSE,
                      metrics=[tf.keras.metrics.MAE])

        # Training: A_train is "frozen" implicitly inside the training algorithm
        history = model.fit(
            self.train_set, self.train_labels,
            validation_data=(self.validation_set, self.validation_labels),
            epochs=training_epochs,
            batch_size=4,
            verbose=1
        )

        # Evaluation on the test set + recompilation to use the custom loss to cope with different dimensions in padding
        print("MODEL EVALUATION - Re-Compiling the Model on test set for evaluation...")
        y_pred_test = model.predict(self.test_set, batch_size=4)
        mask_test = tf.constant(self.create_node_mask(num_nodes_valid=adj_matrix["size"],
                                                      num_nodes_target=adj_matrix["matrix"].shape[0]),
                                dtype=tf.float32)
        # Prediction to take the last step
        # Save the last observation Layer in .npy to have it for prediction
        Y_last_obs = model.predict(self.train_set[-1:])
        # take only the last window-hours, so that you can load this layer easily and use it for prediction
        Y_last_obs = Y_last_obs[:, :, :, -W:]
        print("MODEL TRAINING: Last Observation shape:", Y_last_obs.shape)
        np.save("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name + "\\last_obs_layer.npy", Y_last_obs)

        # Print prediction size to be able to build the prediction framework faster
        print("MODEL EVALUATION - INFO: prediction size on test set: ", y_pred_test.shape)
        loss_test = self.masked_mse(self.test_labels, y_pred_test, mask_test).numpy()
        print("MODEL EVALUATION - MSE on test set: ", loss_test)

        # Save weights and configs into the save directory
        print("MODEL TRAINING - Saving model...")
        model.save_weights("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name + "\\model_weights.weights.h5")
        print("MODEL TRAINING - Model Weights correctly.")
        config = {
            "model_class": "GraphWaveNet",
            "model_user_params": model_params,  # Model params set by the user
            "model_params": {"N": N_train, "F_in": F_in, "W": W, "H": H, "n_blocks": n_blocks},
            "variableToPredict": variableToPredict,
            "end_date": end_date
        }
        # Save config
        with open("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name + "\\model_config.h5", "w") as f:
            json.dump(config, f, indent=4)
        print("MODEL TRAINING - Model config saved correctly.")

        return loss_test