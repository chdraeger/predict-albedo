from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import glob
import pandas as pd
import numpy as np

class dataloader(Sequence):
    """
    Generate data for each batch

    Attributes:
        path: Name of the directory containing the data files
        batch_size: Size of the batch
    """
    def __init__(self, path, batch_size, shuffle=False, to_fit=True, standardize=False, standardize_file=None, transform=False):
        'Initialize'
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.standardize = standardize
        self.standardize_file = standardize_file
        self.transform = transform
        self.data = None
        self.last_file_read = None
        self.samples_pr_file = dict()
        self._batch_file = []
        self._batch_indices_pr_file = []

        count = 0
        filepaths = self.get_all_files(path, shuffle)
        for file in filepaths:
            data = np.load(file)['data']
            size = len(data)
            self.samples_pr_file[file] = size

            indices = np.arange(size)
            if shuffle:
                np.random.shuffle(indices)
            indices_chunk = [indices[x:x+self.batch_size] for x in range(0, len(indices), self.batch_size)]
            self._batch_file += [file] * int(np.ceil(size/batch_size))
            self._batch_indices_pr_file += indices_chunk
            count += size
        self.no_samples = count

    def __len__(self):
        'Denote the number of batches per epoch'
        return len(self._batch_indices_pr_file)

    def get_all_files(self, path, shuffle):
        'Get all files in path'
        filenames = glob.glob(path + '*' + '.npz')
        if shuffle:
            np.random.shuffle(filenames)
        return filenames

    def __getitem__(self, idx):
        'Generate one batch of data'

        file = self._batch_file[idx]
        if file == self.last_file_read:
            data = self.data
        else:
            data = np.load(file)['data']
            if self.standardize:
                std = np.genfromtxt(self.standardize_file, dtype=float, delimiter=',', names=True)
                data = (data - std['mean']) / std['sd']
            self.data = data
            self.last_file_read = file

        X = data[:, :-1]

        # transform to 3-dimensional input (for convolution)
        if self.transform:
            X = X

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
    standardize_file = 'data/meta/std.csv'

    # get data
    train_gen = dataloader('data/train/', batch_size, standardize=True, standardize_file=standardize_file, shuffle=True)
    validate_gen = dataloader('data/validate/', batch_size, standardize=True, standardize_file=standardize_file)
    test_gen = dataloader('data/test/', batch_size, standardize=True, standardize_file=standardize_file)

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
