import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=keras.datasets.fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    
scaled_X_train=X_train/255.0
scaled_y_train=y_train/255.0

def get_model(hidden_layers=1):

  layers=[keras.layers.Flatten(input_shape=(28,28))]
  
  for i in range(hidden_layers):
    layers.append(keras.layers.Dense(500,activation='relu'),)

  layers.append(keras.layers.Dense(10,activation='sigmoid'))

  model=keras.Sequential((layers))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

model = get_model(1)

model.fit(scaled_X_train,y_train,epochs=5)

#model.eveluate()
