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
    def __init__(self, path, batch_size, transform=None, transform_file=None, shuffle=None):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.transform_file = transform_file
        self.filenames = self.get_all_files_in_path(path, shuffle)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size))) # total number of batches in the full dataset

    def get_all_files_in_path(self, path, shuffle):
        filenames = glob.glob(path + '*' + '.RDS')
        if shuffle:
            np.random.shuffle(filenames)
        return filenames

    def __getitem__(self, index): # length of sequence
        file = self.filenames[index]
        data = pyreadr.read_r(file)[None].to_numpy()

        if self.transform:
            std = np.genfromtxt(self.transform_file, dtype=float, delimiter=',', names=True)
            data = (data - std['mean']) / std['sd']

        x = data[:, :-1]
        y = data[:, -1]
        return x, y


# define and build a Sequential model, and print a summary
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
    path_train = 'data/train/'
    path_validate = 'data/validate/'
    transform_file = 'data/meta/std.csv'

    # get data
    train_generator = dataloader(path_train, batch_size, transform=True, transform_file=transform_file, shuffle=True)
    validate_generator = dataloader(path_validate, batch_size, transform=True, transform_file=transform_file)

    # model
    model = build_model()
    build_model().summary()

    callbacks = [ModelCheckpoint(filepath='output/model.hdf5', save_best_only=True),
                 CSVLogger('output/history.csv')]

    fitted_model = model.fit(
        train_generator,
        validation_data=validate_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    # plot
    plt = pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))
    fig = plt.get_figure()
    fig.savefig("output/model.png")
