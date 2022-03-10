# Import modules
import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score


# Load data into a dataframe
dataset = pd.read_csv("admissions_data.csv")

# Print out dataset's columns
print(dataset.columns)

# Print out the first five rows of dataset
print(dataset.head())

# Print out summary statistics of dataset
print(dataset.describe())

# Drop the 'Serial No.' column from dataset
dataset = dataset.drop(["Serial No."], axis=1)

# Create labels dataset (contained in "Chance of Admit") column
labels = dataset.iloc[:, -1]

# Create Features dataset
features = dataset.iloc[:, 0:-1]

# Split data into training set and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=30)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform scaler to the training data
features_train_scaled = scaler.fit_transform(features_train)

# Transform test data instance
features_test_scaled = scaler.transform(features_test)

# Create a neural network model
my_model = Sequential()

# Create an input layer to the model
input = layers.InputLayer(input_shape = (features.shape[1], ))

# Add the input layer to the model (my_model)
my_model.add(input)

# Add hidden layers to the model
my_model.add(layers.Dense(64, activation="relu"))

# Add an ouput layer with one neuron to the model
my_model.add(layers.Dense(1))

# Print out a summary statistics of the model (my_model)
print(my_model.summary())

# Create an instance of Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.01)

# Compile the model
my_model.compile(
  loss="mse",
  metrics=["mae"],
  optimizer=opt
)

# Manual tuning (Adjusting hyperparameters)
stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=40)

# Train the model
history = my_model.fit(features_train_scaled, labels_train, epochs=500, batch_size=30, verbose=1, validation_split=0.2, callbacks=[stop])

# Evaluate the trained model
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test)

# Print out the final loss and the final metric
print(res_mse, res_mae)

# Predict values for features_test_scaled
y_predict = my_model.predict(features_test_scaled)

# Print out the coefficient of determination
print("R-squared value: ",r2_score(labels_test, y_predict))

# Create a figure
fig = plt.figure(figsize=(16,8))

# Create first plot
ax1 = plt.subplot(2,1,1)
plt.plot(history.history["mae"])
plt.plot(history.history["val_mae"])

# Set plot title
ax1.set_title("model mae")
# Set y-axis label
ax1.set_ylabel("MAE")
# Set x-axis label
ax1.set_xlabel("epoch")
# Set plot legend
ax1.legend(["train", "validation"], loc="upper left")

# Create second plot
ax2 = plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# Set plot title
ax2.set_title("model loss")
# Set y-axis label
ax2.set_ylabel("loss")
# Set x-axis label
ax2.set_xlabel("epoch")
# Set plot legend
ax2.legend(["train", "validation"], loc="upper left")

# Adjust spaces between the two plots
plt.subplots_adjust(top=0.95, hspace=0.35)

# Show plot
plt.show()

# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

fig.savefig('static/images/my_plots.png')