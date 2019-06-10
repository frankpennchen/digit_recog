import os
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

class cnn_solution(object):
    def __init__(self):

        self.path_train = './Data/train.csv'
        self.path_test = './Data/test.csv'

        self.train_df_raw = pd.read_csv(self.path_train)
        self.train_df = self.train_df_raw.copy()
        
        self.X_train = self.train_df.drop(['label'], 1)
        self.Y_train = self.train_df['label']
        
        # Add 2 dimensions to pass a 4D tensor to the CNN and normalize values
        self.X_train = self.X_train.values.reshape(self.X_train.shape[0], 28, 28, 1).astype('float32') / 255 
        # One-hot encoding of target
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, 10)

    def build_cnn(self):
        
        model = tf.keras.models.Sequential()
        
        # Multiple convolution operations to detect features in the images
        model.add(tf.keras.layers.Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32,kernel_size=3,activation='relu')) # no need to specify shape as there is a layer before
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4)) # reduce overfitting
    
        model.add(tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4)) # reduce overfitting
        
        # Flattening and classification by standard ANN
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
        
        return model
    
    def train(self):

        X_test = pd.read_csv(self.path_test)
        X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
        
        model = self.build_cnn()
        model.fit(self.X_train, self.Y_train, batch_size=64, epochs=16)
        prediction = model.predict_classes(X_test, verbose=0)
        submission = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),
                                 "Label": prediction})
        submission.to_csv("submission.csv", index=False, header=True)


# ============================================================================

Solution1 = cnn_solution()
Solution1.train()






