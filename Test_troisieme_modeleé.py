import pandas as pd
import numpy as np
import os,typing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers,models
from normalizing_data import preprocess
from modif_dataset import modif, recup_data_avenue
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
import random


dataset = r"\dataset_paris.csv"

scale = 1.0 # de base 0.6
train_prop = .6 #de base 0.8
sequence_len = 16 #de base 16
batch_size = 32 #de base 32
epochs = 100 #de base 100
features = ["Taux d'occupation"]
features_len = 24 #de base 24

#data = modif(dataset)
data = recup_data_avenue(dataset)



#on utilise les proportions choisies
data = data[:int(scale*len(data))]
train_len = int(train_prop*len(data))


#on crée les dataset d'entrainements et de test
dataset_train = data[:train_len]
dataset_test = data[train_len:]

#on normalise
mean = dataset_test.mean()
std = dataset_train.std()

dataset_train = (dataset_train - mean)/std
dataset_test = (dataset_test - mean) / std


#dataset_train = dataset_train.to_numpy()
#dataset_test = dataset_test.to_numpy()



train_generator = TimeseriesGenerator(dataset_train, dataset_train, length=sequence_len, batch_size=batch_size)
test_generator = TimeseriesGenerator(dataset_test, dataset_test, length=sequence_len, batch_size=batch_size)


#Creation du modele

model = Sequential()
model.add(keras.layers.InputLayer(input_shape= (sequence_len,24)))
model.add(keras.layers.LSTM(100, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(24))
model.summary()


model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])


#entrainenement du model
history = model.fit(train_generator,
                    epochs=epochs,
                    verbose=2,
                    validation_data= test_generator,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])


historique = pd.DataFrame(history.history)
#figure, axe = plt.subplots(figsize = (14,8))
#num_epoque = historique.shape[0]
#axe.plot(np.arange(0, num_epoque),historique["mae"], label = "Training mae", lw = 3, color = 'red')
#axe.plot(np.arange(0, num_epoque),historique["val_mae"], label = "Validation mae", lw = 3, color = 'blue')
#axe.legend ()
#plt.tight_layout()
#plt.show()


def predict():
    for p in range(3):
        s = random.randint(0,len(dataset_train)-sequence_len)

        sequence = dataset_test[s:s + sequence_len]
        sequence_true = dataset_test[s: s + sequence_len + 1 ]

        pred = model.predict(np.array([sequence]))


        def denormalize(seq):
            nseq = seq.copy()
            for i,s in enumerate(nseq):
                s = s*std + mean
                nseq[i] = s
            return nseq

        prediction = denormalize(pred)
        real = denormalize(sequence_true[-1])
        real_1 = denormalize(sequence_true[-2])
        real_2 = denormalize(sequence_true[-3])
        real_3 = denormalize(sequence_true[-4])

        #on récupère les quelques jours avant pour comparer
        sequence_real = [real, real_1, real_2, real_3]

        hours = np.arange(1,25)

        print(prediction, real)

        plt.plot(hours,prediction[0], "o-", label = "Prédictions")
        for k, verite in enumerate(sequence_real):
            plt.plot(hours, verite, "o-", label = f"Valeurs réelles : J-{k}")

        plt.legend()
        plt.title(f"{features[0]}, Paramètres : epochs = {epochs} , sequence_len = {sequence_len}, batch_size = {batch_size}")
        plt.show()


predict()
#scale = 0.8 # de base 0.6
#train_prop = .8 #de base 0.8
#sequence_len = 4 #de base 16
#batch_size = 16 #de base 32
#epochs = 24 #de base 100
#features = ["Taux d'occupation"]
#features_len = 24 #de base 24
#
