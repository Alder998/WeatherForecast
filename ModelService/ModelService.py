# This class is to create a Neural Network Library Built upon TensorFlow

import numpy as np
from keras.src.layers import TimeDistributed, MaxPooling1D, Conv1D, Flatten, Reshape, UpSampling1D
from sklearn.preprocessing import StandardScaler
import DataPreparation as dt
import tensorflow as tf
import joblib

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
            joblib.dump(scaler, 'scaler_labels_' + model_name + '.pkl')

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
        if len(modelStructure['Conv']) > 0:
            for c in range(len(modelStructure['Conv'])):
                units = modelStructure['Conv'][c]
                model.add(Conv1D(filters=units, kernel_size=3, activation='relu', padding='same'))
                model.add(MaxPooling1D(pool_size=2, padding='same'))
                # Upsampling to preserve the dimensionality
                model.add(UpSampling1D(size=2))

        # Proceed with the Recurrent Part with LSTM layers
        if len(modelStructure['LSTM']) > 0:
            # Expand dimensions to add timestamp
            for l in range(len(modelStructure['LSTM'])):
                units = modelStructure['LSTM'][l]
                if l == 0:
                    # Input_shape required on the first layer
                    layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True,
                                                 input_shape=(None, train_set.shape[2]))
                else:
                    if return_seq_last_rec_layer:
                        # For others, no issues
                        # Remove the timestamp param if the layer is the last one of the LSM Structure
                        if l == len(modelStructure['LSTM']) - 1:
                            layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=False)
                        else:
                            layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True)
                    else:
                        layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True)
                # Add the dropout layer
                model.add(tf.keras.layers.Dropout(dropout_LSTM))
                # Finally, add the layer to the model
                model.add(layer)

        # then, add the FF layer
        for l in range(len(modelStructure['FF'])):
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

        # Now, Train the Model
        model.fit(train_set, train_labels, epochs=trainingEpochs)

        # Evaluate on Test set
        test_loss, test_acc = model.evaluate(test_set, test_labels, verbose=2)
        print('Test Mean-Squared-Error:', '{:,}'.format(test_acc))

        # Save the model il .h5 format
        model.save(save_name + '.h5')
        print('Model Saved Correctly!')

        return model




