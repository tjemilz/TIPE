import keras.optimizers
import numpy as np
from keras.layers import Dense, ConvLSTM2D, Dropout, Flatten,RepeatVector, Bidirectional, LSTM,TimeDistributed
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt



model = Sequential()

model.add(ConvLSTM2D(filters= 128, kernel_size=(1,4) ,
                     activation= 'relu',
                     return_sequences= True,
                     input_shape=((4,))
                    ))
model.add(Dropout(0.2))
model.add(ConvLSTM2D(filters= 128, kernel_size=(1,4) ,
                     activation= 'relu',
                     return_sequences= True,
                     input_shape=((4,))
                    ))
model.add(Dropout(0.2))
model.add(ConvLSTM2D(filters= 128, kernel_size=(1,4) ,
                     activation= 'relu',
                     return_sequences= True,
                     input_shape=((4,))
                    ))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(RepeatVector(1))
model.add(Bidirectional(LSTM(512,
                             activation = 'relu',
                             return_sequences = True),
                        input_shape= (4,)
                        ))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss= 'mse', optimizer=keras.optimizers.Adam(lr=0.0001))

training = model.fit(x, y, epochs=300, validation_split=0.2, batch_size=5)


_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

historique = pd.DataFrame(training.history)
historique['epoque'] = training.epoch
figure, axe = plt.subplots(figsize = (14,8))
num_epoque = historique.shape[0]
axe.plot(np.arange(0, num_epoque),historique["mae"], label = "Training MAE", lw = 3, color = 'red')
axe.plot(np.arange(0, num_epoque),historique["val_mae"], label = "Validation MAE", lw = 3, color = 'blue')
axe.legend ()
plt.tight_layout()
plt.show()