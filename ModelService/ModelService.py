# This class is to create a Neural Network Library Built upon TensorFlow
import numpy as np
import DataPreparation as dt
import tensorflow as tf

class ModelService:

    def __init__(self, train_set, test_set, train_labels, test_labels):
        self.train_set = train_set
        self.test_set = test_set
        self.train_labels = train_labels
        self.test_labels = test_labels
        pass

    def NNModel (self, modelStructure, trainingEpochs):

        # Adapt data for Model
        train_set = np.stack(self.train_set, axis=0)
        test_set = np.stack(self.test_set, axis=0)
        train_labels = np.stack(self.train_labels, axis=0)
        test_labels = np.stack(self.test_labels, axis=0)

        # Make the array numeric
        train_set = np.array(train_set, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        test_set = np.array(test_set, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.float32)

        print(train_set.dtype)
        print(train_labels.dtype)

        print(train_set.shape)
        print(train_labels.shape)

        # Instatiate The Tensorflow Object Model
        model = tf.keras.Sequential()

        if len(modelStructure['LSTM']) > 0:
            # Expand dimensions to add timestamp
            #model.add(tf.keras.layers.Reshape((1, train_set.shape[1]), input_shape=(train_set.shape[1],)))
            for l in range(len(modelStructure['LSTM'])):
                units = modelStructure['LSTM'][l]
                if l == 0:
                    # Input_shape required on the first layer
                    layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True,
                                                 input_shape=(None, train_set.shape[1]))
                else:
                    # For others, no issues
                    # Remove the timestamp param if the layer is the last one of the LSM Structure
                    if l == len(modelStructure['LSTM']) - 1:
                        layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=False)
                    else:
                        layer = tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True)
                model.add(layer)
            # then, add the FF layer
        for l in range(len(modelStructure['FF'])):
            unitsFF = modelStructure['FF'][l]
            layerFF = tf.keras.layers.Dense(unitsFF, activation='relu')
            model.add(layerFF)
        model.add(tf.keras.layers.Dense(2))

        # Compile
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        # Now, Train the Model
        model.fit(train_set, train_labels, epochs=trainingEpochs)

        # Evaluate on Test set
        test_loss, test_acc = model.evaluate(test_set, test_labels, verbose=2)
        print('Test accuracy:', test_acc)



