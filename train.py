from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import glob
import pandas as pd
import numpy as np
import pyreadr

class dataloader(Sequence):
    def __init__(self, path, batch_size, transform=False, transform_file=None, shuffle=False, to_fit=True):
        self.path = path
        self.batch_size = batch_size
        self.transform = transform
        self.transform_file = transform_file
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.filenames = self.get_all_files(path, shuffle)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def get_all_files(self, path, shuffle):
        'Get all files in path'
        filenames = glob.glob(path + '*' + '.RDS')
        if shuffle:
            np.random.shuffle(filenames)
        return filenames

    def __getitem__(self, index): # index: length of sequence
        'Generate one batch of data'
        file = self.filenames[index]
        data = pyreadr.read_r(file)[None].to_numpy()

        if self.transform:
            std = np.genfromtxt(self.transform_file, dtype=float, delimiter=',', names=True)
            data = (data - std['mean']) / std['sd']

        X = data[:, :-1]

        if self.to_fit:
            y = data[:, -1]
            return X, y
        else:
            return X


# define and build a Sequential model
def build_model():
    model = Sequential()
    model.add(layers.Dense(2**8, input_shape=(22,), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2**7, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation=None))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


if __name__ == "__main__":
    # specifications
    epochs = 10 # 30
    batch_size = 40000 # 2**7
    transform_file = 'data/meta/std.csv'

    # get data
    train_gen = dataloader('data/train/', batch_size,
                                 transform=True, transform_file=transform_file, shuffle=True)
    validate_gen = dataloader('data/validate/', batch_size,
                                    transform=True, transform_file=transform_file)
    test_gen = dataloader('data/test/', batch_size,
                                transform=True, transform_file=transform_file, to_fit=True)

    # fit model
    model = build_model()
    build_model().summary()

    callbacks = [ModelCheckpoint(filepath='output/model.hdf5', save_best_only=True),
                 CSVLogger('output/history.csv')]

    fitted_model = model.fit(
        train_gen,
        validation_data=validate_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # plot training + validation
    plt = pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))
    fig = plt.get_figure()
    fig.savefig("output/model.png")

    # predict on test set
    prediction = model.evaluate(test_gen)
    print("test loss, test mae:", prediction)
