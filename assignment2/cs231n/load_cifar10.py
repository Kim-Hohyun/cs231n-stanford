import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10  

def get_cifar10_data(channel_first=False):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    num_training = 49000
    num_validation = 1000
    num_test = 1000

    # subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask].astype('float64')
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask].astype('float64')
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask].astype('float64')
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    if channel_first:
        X_train = X_train.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)
        X_test = X_test.transpose(0, 3, 1, 2)
        
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    data = {
          'X_train': X_train,
          'y_train': y_train,
          'X_val': X_val,
          'y_val': y_val,
          'X_test': X_test,
          'y_test': y_test
    }
    return data