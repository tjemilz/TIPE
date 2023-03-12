import numpy as np

def preprocess(data_array : np.ndarray, train_size,val_size):
    num_times_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_times_steps * train_size),
        int(num_times_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis = 0), train_array.std(axis = 0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train: (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val):] - mean) / std

    return train_array, val_array, test_array


