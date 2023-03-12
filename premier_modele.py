import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import loadtxt
from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd
from modif_dataset import modif


dataset = r"C:\Users\efour\Partage_PC\Cours\TIPE\MP\Dataset\dataset_paris.csv"
#dataset_test = loadtxt(r"C:\Users\efour\Partage_PC\Cours\TIPE\MP\Dataset\dataset_test.csv", delimiter=',')
train_prop =.8
data = modif(dataset)

train_len = int(len(data)*train_prop)

dataset_train = data[:train_len]
dataset_test = data[train_len:]

#x_test = dataset_test[:, 0:4]
#y_test = dataset_test[:, 5]

model = Sequential()
model.add(Dense(256, input_shape=(24,), activation='relu'))
model.add(Dropout(0.3, seed=2))
model.add(Dense(256, activation='swish'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='swish'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='swish'))
model.add(Dense(24))

# compile the keras model
model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

#mae = mean absolute error (erreur absolue moyenne)


# fit the keras model on the dataset
training = model.fit(dataset_train,dataset_test, epochs=10, validation_split=0.2, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(dataset_train, dataset_test)
print('Accuracy: %.2f' % (accuracy*100))

historique = pd.DataFrame(training.history)
historique['epoque'] = training.epoch
figure, axe = plt.subplots(figsize = (14,8))
num_epoque = historique.shape[0]
axe.plot(np.arange(0, num_epoque),historique["accuracy"], label = "Training MAE", lw = 3, color = 'red')
axe.plot(np.arange(0, num_epoque),historique["val_accuracy"], label = "Validation MAE", lw = 3, color = 'blue')
axe.legend()
plt.tight_layout()
plt.show()


