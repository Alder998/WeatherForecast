# This class is to create a Neural Network Library Built upon TensorFlow
import json
import os

import numpy as np
from keras.src.layers import TimeDistributed, MaxPooling1D, Conv1D, Conv2D, Flatten, Reshape, UpSampling1D, \
    UpSampling2D, MaxPooling2D, GlobalAveragePooling2D, Permute, GlobalAveragePooling1D, InputLayer, Bidirectional
from sklearn.preprocessing import StandardScaler
import DataPreparation as dt
import tensorflow as tf
import joblib
from ModelStorageService import ModelStorageService as st

class ModelService:

    def __init__(self, train_set, test_set, train_labels, test_labels):
        self.train_set = train_set
        self.test_set = test_set
        self.train_labels = train_labels
        self.test_labels = test_labels
        pass

    # Utils-like function to standardize the data
    def standardizeData(self, set, saveScaler=False, model_name='None'):

        # Initialize the scaler from scikit-learn
        scaler = StandardScaler()

        # Reshape that is necessary to process the features separately
        num_samples, num_obs, num_features = set.shape
        X_reshaped = set.reshape(-1, num_features)

        # Normalize the features
        X_scaled = scaler.fit_transform(X_reshaped)

        # Put the array in the old shape
        X_scaled = X_scaled.reshape(num_samples, num_obs, num_features)

        if saveScaler:
            # Appropriate Path to save the scaler
            scaler_savePath = ("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + model_name)
            joblib.dump(scaler, scaler_savePath + '\\scaler_labels_' + model_name + '.pkl')

        return X_scaled

    def NNModel (self, modelStructure, trainingEpochs, save_name, dropout_LSTM, dropout_FF,
                 return_seq_last_rec_layer = False, standardize=True):

        # Adapt data for Model
        train_set = np.stack(self.train_set, axis=0)
        test_set = np.stack(self.test_set, axis=0)
        train_labels = np.stack(self.train_labels, axis=0)
        test_labels = np.stack(self.test_labels, axis=0)

        if standardize:
            # Standardize the data
            train_set = self.standardizeData(train_set)
            # Save the scaler only for train (test set may have fewer observations)
            train_labels = self.standardizeData(train_labels, saveScaler=True, model_name=save_name)
            test_set = self.standardizeData(test_set)
            test_labels = self.standardizeData(test_labels)

        # Make the array numeric
        train_set = np.array(train_set, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        test_set = np.array(test_set, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)

        print('Train Set shape:', train_set.shape)
        print('Train Labels shape:', train_labels.shape)
        print('Test Set shape:', test_set.shape)
        print('Test Labels shape:', test_labels.shape)

        # Instatiate The Tensorflow Object Model
        model = tf.keras.Sequential()

        # Start with the CNN, to understand the space dependencies
        if len(modelStructure['Conv1D']) > 0:
            for c in range(len(modelStructure['Conv1D'])):
                units = modelStructure['Conv1D'][c]
                model.add(Conv1D(filters=units, kernel_size=3, activation='relu', padding='same', input_shape=(train_set.shape[1], train_set.shape[2])))
                model.add(MaxPooling1D(pool_size=2, padding='same'))
                # Upsampling to preserve the dimensionality
                model.add(UpSampling1D(size=2))

        # Add the Con2D layer
        if len(modelStructure['Conv2D']) > 0:
            #train_set = tf.expand_dims(train_set, axis=-1)  # (timeSteps, batch, features, 1)
            #train_labels = tf.expand_dims(train_labels, axis=-1)  # (timeSteps, batch, features, 1)
            for c in range(len(modelStructure['Conv2D'])):
                units = modelStructure['Conv2D'][c]
                if c == 0:
                    # (1368, 604, 6) must be input_shape=(604, 6, 1) or input_shape=(604, 1, 6)
                    model.add((Conv2D(filters=units, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same", activation="relu")))
                else:
                    model.add((Conv2D(filters=units, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same", activation="relu")))
                model.add((MaxPooling2D(pool_size=(2, 2), padding='same')))
                # Upsampling to preserve the dimensionality
                model.add((UpSampling2D(size=(2, 2))))

            # This layer will only be used for time-space, so it must be RESHAPED to fit with the
            # LSTM layer, that by default needs 3-dimensional inputs, therefore:
            # Reshape: (batch, H, W, C) → (batch, H*W, C)
            #model.add(Reshape((train_set.shape[2], -1)))  # keeps time dimension
            #model.add((Reshape((-1,))))
            #model.add(TimeDistributed(Flatten()))  # (batch, time_steps, features)
            #model.add(Permute((2, 1)))
            #model.add(Reshape((train_set.shape[1], -1)))

            # Permute to (timesteps, batch, features)
            #model.add(Permute((2, 1)))  # now shape: (timesteps, batch, features)
            #model.add(Permute((2, 1)))  # back to (batch, timesteps, features)

        # Proceed with the Recurrent Part with LSTM layers
        if len(modelStructure['LSTM']) > 0:
            # Expand dimensions to add timestamp
            for l in range(len(modelStructure['LSTM'])):
                units = modelStructure['LSTM'][l]
                if l == 0:
                    # Input_shape required on the first layer
                    layer = Bidirectional(tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True,
                                                 input_shape=(None, train_set.shape[2])))
                else:
                    if return_seq_last_rec_layer:
                        # For others, no issues
                        # Remove the timestamp param if the layer is the last one of the LSM Structure
                        if l == len(modelStructure['LSTM']) - 1:
                            layer = Bidirectional(tf.keras.layers.LSTM(units, activation='tanh', return_sequences=False))
                        else:
                            layer = Bidirectional(tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True))
                    else:
                        layer = Bidirectional(tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True))
                # Add the dropout layer
                model.add(tf.keras.layers.Dropout(dropout_LSTM))
                # Finally, add the layer to the model
                model.add(layer)

        # Add the Conv2DLSTM layer, if needed
        if len(modelStructure['Conv2DLSTM']) > 0:
            for l in range(len(modelStructure['LSTM'])):
                units = modelStructure['LSTM'][l]
                if l == 0:
                    layer = tf.keras.layers.ConvLSTM2D(
                        filters=units, kernel_size=(3, 3), strides=(1, 1),
                        padding="same", activation="tanh", return_sequences=True,
                        input_shape=(None, train_set.shape[2]))
                else:
                    if l == len(modelStructure['Conv2DLSTM']) - 1:
                        layer = tf.keras.layers.ConvLSTM2D(
                            filters=units, kernel_size=(3, 3), strides=(1, 1),
                            padding="same", activation="tanh", return_sequences=False)
                    else:
                        layer = tf.keras.layers.ConvLSTM2D(
                            filters=units, kernel_size=(3, 3), strides=(1, 1),
                            padding="same", activation="tanh", return_sequences=True)
                # Add the dropout layer
                model.add(tf.keras.layers.Dropout(dropout_LSTM))
                model.add(layer)

        # then, add the FF layer
        for l in range(len(modelStructure['FF'])):
            if (len(modelStructure['Conv2D']) != 0) & (l==0):
                # Added the global average pooling layer to handle the flattening for the FF layer
                model.add(GlobalAveragePooling1D())
            unitsFF = modelStructure['FF'][l]
            layerFF = tf.keras.layers.Dense(unitsFF, activation='relu')
            # Add the Dropout layer for FF
            model.add(tf.keras.layers.Dropout(dropout_FF))
            # Finally, add the FF layer to the model
            model.add(layerFF)
        # This layer is to correct the Mismatch of size caused by the Up-sampling of the Convolutional Layer
        # Dense final Layer for regression
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # Compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['mse'])

        # Model.summary() to see the model shapes
        #print(model.summary())

        # Now, Train the Model
        if len(modelStructure['Conv2D']) != 0:
            #train_set = tf.expand_dims(train_set, axis=-1)  # (timeSteps, batch, features, 1)
            #train_labels = tf.expand_dims(train_labels, axis=-1)  # (timeSteps, batch, features, 1)
            model.fit(tf.expand_dims(train_set, axis=-1), tf.expand_dims(train_labels, axis=-1), epochs=trainingEpochs)
            # Evaluate on Test set
            test_loss, test_acc = model.evaluate(tf.expand_dims(test_set, axis=-1), tf.expand_dims(test_labels, axis=-1), verbose=2)
            print('Test Mean-Squared-Error:', '{:,}'.format(test_acc))
        else:
            model.fit(train_set, train_labels, epochs=trainingEpochs)
            # Evaluate on Test set
            test_loss, test_acc = model.evaluate(test_set, test_labels, verbose=2)
            print('Test Mean-Squared-Error:', '{:,}'.format(test_acc))

        # Save the model in .h5 format in the appropriate folder
        model.save("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + save_name + "\\" + save_name + '.h5')
        print('Model Saved Correctly!')

        # Save info
        st.ModelStorageService(save_name).saveModelInfo(NN_structure=modelStructure, epochs=trainingEpochs,
                                                        training_test_shape=train_set.shape, test_set_shape=test_set.shape,
                                                        test_loss=test_acc)

        return model

    # Function to continue model training from a saved model, and save it again
    def continueModelTraining (self, modelName, newTrainingEpochs, standardize=True):

        # Extract the number of epochs from the JSON name
        # Read the JSON file
        folder_path = "D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + modelName + '\\modelInfo.json'
        with open(folder_path, "r") as f:
            model_info = json.load(f)

        # Process existing Epochs and new Epochs
        existingEpochs = model_info["Training Epochs"]
        print('Existing Epochs: ' + str(existingEpochs))
        newEpochs = int(existingEpochs) + newTrainingEpochs

        # Process the data
        # Adapt data for Model
        train_set = np.stack(self.train_set, axis=0)
        test_set = np.stack(self.test_set, axis=0)
        train_labels = np.stack(self.train_labels, axis=0)
        test_labels = np.stack(self.test_labels, axis=0)

        if standardize:
            # Standardize the data
            train_set = self.standardizeData(train_set)
            # Save the scaler only for train (test set may have fewer observations)
            train_labels = self.standardizeData(train_labels, saveScaler=True, model_name=modelName)
            test_set = self.standardizeData(test_set)
            test_labels = self.standardizeData(test_labels)

        # Make the array numeric
        train_set = np.array(train_set, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        test_set = np.array(test_set, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)

        print('Train Set shape:', train_set.shape)
        print('Train Labels shape:', train_labels.shape)
        print('Test Set shape:', test_set.shape)
        print('Test Labels shape:', test_labels.shape)

        # Load stored model
        existingModel = tf.keras.models.load_model("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + modelName + '\\' + modelName + ".h5", compile=False)

        existingModel.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['mse'])

        # Continue the model training
        existingModel.fit(train_set, train_labels, epochs=newTrainingEpochs)

        # Evaluate on Test set
        test_loss, test_acc = existingModel.evaluate(test_set, test_labels, verbose=2)
        print('Test Mean-Squared-Error:', '{:,}'.format(test_acc))

        # Save the model il .h5 format, overwriting the existing one
        existingModel.save("D:\\PythonProjects-Storage\\WeatherForecast\\Stored-models\\" + modelName + "\\" + modelName + '.h5')
        print('Model Saved Correctly!')

        # Now, modify the model info that need to be updated, i.e. The Epochs, train, test shape, test loss
        model_info["Training Epochs"] = newEpochs
        model_info["Train set shape"] = train_set.shape
        model_info["Test set shape"] = test_set.shape
        model_info["Test loss"] = test_acc

        data_str_keys = {str(k): v for k, v in model_info.items()}
        # save the JSON
        with open(folder_path, "w") as f:
            json.dump(data_str_keys, f, indent=None, separators=(",", ": ") )
            print('Model Info saved correctly!')

        return existingModel








