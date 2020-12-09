### Script to implement a standard 3D CNN, train & save the model and run predictions

import numpy as np
import numpy.random as rng
import tqdm
import scipy as sp
import scipy.constants as spc
import tensorflow as tf

data = np.load("flat_train_set_Normalized_h0_175_final_4Mpc.npz")

y_train = data["M200c_list"]
x_train = data["phase_space_3D_KDE"]

y_train = np.array(y_train)
x_train = np.array(x_train)

constant_scaling = 1000
x_train *= constant_scaling

print(x_train.shape)
print(y_train.shape)

train_size = y_train.size


neural_net = tf.keras.Sequential([
    tf.keras.layers.Convolution3D(12,
                             kernel_size=5,
                             padding="VALID",
                             strides=[1, 1, 1],
                             activation=tf.nn.relu,
                             input_shape=(50,50,50,1)),
    tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2],
                                 strides=[2, 2, 2],
                                 padding="SAME"),
    tf.keras.layers.Convolution3D(8,
                             kernel_size=3,
                             padding="VALID",
                             strides=[1, 1, 1],
                             activation=tf.nn.relu),
    tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2],
                                 strides=[2, 2, 2],
                                 padding="SAME"),
    tf.keras.layers.Convolution3D(4,
                             kernel_size=1,
                             padding="VALID",
                             strides=[1, 1, 1],
                             activation=tf.nn.relu),
    tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2],
                                 strides=[2, 2, 2],
                                 padding="SAME"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1),
    ])


x_train = np.reshape(x_train, [train_size,50,50,50,1])

data = np.load("test_set_Normalized_h0_175_final_4Mpc.npz")

y_val = data["M200c_list"]
x_val = data["phase_space_3D_KDE"]

y_val = np.array(y_val)
x_val = np.array(x_val)

x_val *= constant_scaling

x_val = np.reshape(x_val, [y_val.size,50,50,50,1])

neural_net.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.00005), loss="mean_squared_error")

print(neural_net.summary())

from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 10,
                          verbose = 1,
                          restore_best_weights = True)

hist = neural_net.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_val, y_val), callbacks=[earlystop], verbose=True)


### TEST SET

data = np.load("test_set_Normalized_h0_175_final_4Mpc.npz")

y_test = data["M200c_list"]
x_test = data["phase_space_3D_KDE"]

y_test = np.array(y_test)
x_test = np.array(x_test)

ground_truth = []
prediction = []

x_test *= constant_scaling

for k in range(y_test.size):
    pred_index = int(k)
    
    y_pred = neural_net(x_test[pred_index,:,:,:].reshape(1,50,50,50,1))
    y_pred = y_pred.numpy()[0,0]
    prediction.append(y_pred)
    
    ground_truth.append(y_test[pred_index])

np.savez("predictions/CNN_3D_predictions_test_h0_175_final_4Mpc.npz", ground_truth=ground_truth, prediction=prediction)


### EVALUATION SET

print("Running predictions for clusters in evaluation set...")

data = np.load("validation_set_Normalized_h0_175_final_4Mpc.npz")

y_val = data["M200c_list"]
x_val = data["phase_space_3D_KDE"]

y_val = np.array(y_val)
x_val = np.array(x_val)

x_val *= constant_scaling

ground_truth = []
prediction = []

for k in range(y_val.size):
    pred_index = int(k)

    y_pred = neural_net(x_val[pred_index,:,:,:].reshape(1,50,50,50,1))
    y_pred = y_pred.numpy()[0,0]
    prediction.append(y_pred)

    ground_truth.append(y_val[pred_index])

np.savez("predictions/CNN_3D_predictions_validation_h0_175_final_4Mpc.npz", ground_truth=ground_truth, prediction=prediction)


### SDSS-III set

print("Running predictions for SDSS clusters...")

data = np.load("SDSS_Normalized.npz")
Num_SDSS = 909

x_sdss = data["phase_space_3D_KDE"]
x_sdss = np.array(x_sdss)

x_sdss *= constant_scaling

prediction = []

for k in range(Num_SDSS):
    pred_index = int(k)

    y_pred = neural_net(x_sdss[pred_index,:,:,:].reshape(1,50,50,50,1))
    y_pred = y_pred.numpy()[0,0]
    prediction.append(y_pred)

np.savez("predictions/CNN_3D_predictions_SDSS_final_transform_4Mpc.npz", prediction=prediction)


### Save trained 3D CNN

neural_net.save("models/CNN_3D_mass_estimator.h5")

print("Model saved and script executed successfully!")
