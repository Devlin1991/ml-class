from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# is_give_train stores a boolean value for each index of y_train, true if the value is 5
is_five_train = y_train == 5

# same as above but for the test data
is_five_test = y_test == 5

# labels array to act as a conversion of the boolean is_five_* values into human readable labels
labels = ["Not Five", "Is Five"]

# Stores the dimensions of the training data images which are stored in index 1 and 2 of X_train
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model = Sequential()
# Adds a layer to the model that is the size of flattened 2D image to 1D
model.add(Flatten(input_shape=(img_width, img_height)))
# 
model.add(Dense(1, activation="sigmoid"))

# Set loss to Mean Absolute Error, use adam optimizer, metric is accuracy
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])

# Fit the model (Training)
model.fit(X_train, is_five_train, epochs=5, validation_data=(X_test, is_five_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])

# Output trained model to file
model.save('perceptron.h5')
