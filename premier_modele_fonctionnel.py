import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import loadtxt
from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd


dataset_train = loadtxt(r"C:\Users\efour\Partage_PC\Cours\TIPE\MP\Dataset\dataset_train.csv", delimiter=',')
dataset_test = np.array([[364,480,5.4,20,65,1]])

# les données sont [

x = dataset_train[:, 0:4]
y = dataset_train[:, 5]

x_test = dataset_test[:, 0:4]
y_test = dataset_test[:, 5]

model = Sequential()
model.add(Dense(64, input_shape=(4,), activation='swish'))
model.add(Dropout(0.3, seed=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(5))

# compile the keras model
model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

#mae = mean absolute error (erreur absolue moyenne)


# fit the keras model on the dataset
training = model.fit(x, y, epochs=20, validation_split=0.2, batch_size=5)

# evaluate the keras model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

historique = pd.DataFrame(training.history)
historique['epoque'] = training.epoch
figure, axe = plt.subplots(figsize = (14,8))
num_epoque = historique.shape[0]
axe.plot(np.arange(0, num_epoque),historique["accuracy"], label = "Training Accuracy", lw = 3, color = 'red')
axe.plot(np.arange(0, num_epoque),historique["val_accuracy"], label = "Validation Accuracy", lw = 3, color = 'blue')
axe.legend ()
plt.tight_layout()
plt.show()


model.predict(x_test)
y_pred = model.predict(x_test)
print(y_test)
print(y_pred)