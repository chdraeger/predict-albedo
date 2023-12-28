from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import glob
import pandas as pd
import numpy as np
from pathlib import Path

class dataloader(Sequence):
    """
    Generate data for each batch

    Attributes:
        path: Name of the directory containing the data files
        batch_size: Size of the batch
    """
    def __init__(self, path, batch_size, shuffle=False, to_fit=True, standardize=False, standardize_file=None, transform=False):
        """
        Initialize
        """
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

        # get lists of length [nr of batches] with the file name and the relevant indices per batch
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
        """
        Denote the number of batches per epoch
        """
        return len(self._batch_indices_pr_file)

    def get_all_files(self, path, shuffle):
        """
        Get all files in path

        :param path:
        :param shuffle:
        :return:
        """
        filenames = glob.glob(path + '*' + '.npz')
        if shuffle:
            np.random.shuffle(filenames)
        return filenames

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """

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
def build_model(model_type = 'fnn'):

    match model_type:
        case 'fnn':
            model = Sequential()
            model.add(layers.Dense(2 ** 8, input_shape=(22,), activation='relu'))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(2 ** 7, activation='relu'))
            model.add(layers.Dropout(0.1))
            model.add(layers.Dense(1, activation=None))
            model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        case 'lstm':
            model = Sequential()
        case 'resnet':
            model = Sequential()

    return model


if __name__ == "__main__":
    # specifications
    epochs = 2 # 30
    batch_size = 40000 # 2**7
    standardize_file = 'data/meta/std.csv'
    transform = False
    model_type = 'fnn'

    print('Initiate data generators \n')
    train_gen = dataloader('data/train1/', batch_size, standardize=True, standardize_file=standardize_file, transform=transform, shuffle=True)
    validate_gen = dataloader('data/validate1/', batch_size, standardize=True, standardize_file=standardize_file, transform=transform)
    test_gen = dataloader('data/test1/', batch_size, standardize=True, standardize_file=standardize_file, transform=transform)

    print('Fit model \n')
    model = build_model(model_type=model_type)
    build_model().summary()

    # checkpoints
    output_path = 'output_' + model_type + '/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    callbacks = [ModelCheckpoint(filepath=output_path + 'model.keras', save_best_only=True),
                 CSVLogger(output_path + 'history.csv')]

    fitted_model = model.fit(
        train_gen,
        validation_data=validate_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    print('Plot training and validation \n')
    plt = pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))
    fig = plt.get_figure()
    fig.savefig(output_path + "loss_train_val.png")

    print('Predict on test set \n')
    prediction = model.evaluate(test_gen)
    np.savetxt(output_path + 'loss_test.txt', prediction, header="test_loss,test_mae")
