import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tensorflow import keras
import CONSTANTS

class dataloader(keras.utils.Sequence):
    def __init__(self, path, batch_size, shuffle=False, to_fit=True,
                 standardize=False, standardize_file=None, transform=False):
        """
        Initialize

        :param path:
        :param batch_size:
        :param shuffle:
        :param to_fit:
        :param standardize:
        :param standardize_file:
        :param transform:
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
        self.no_samples = count # total number of samples

    def __len__(self):
        """
        Denote the number of batches per epoch.

        :return:
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

        :param idx:
        :return:
        """

        file = self._batch_file[idx]
        if file == self.last_file_read:
            data = self.data
        else:
            data = np.load(file)['data']
            self.data = data
            self.last_file_read = file

        indices = self._batch_indices_pr_file[idx]
        data = data[indices]

        X = np.delete(data, CONSTANTS.OUTPUT, axis=1)
        if self.standardize:
            std = np.genfromtxt(self.standardize_file, dtype=float, delimiter=',', names=True)
            X = (X - std['mean']) / std['sd']

        # transform to 3-dimensional input (for convolution) with time as the third dimension
        # read dimensions from meta file
        # dimensions: number of samples = batch size, sequence length = 6 (time), features = 12
        if self.transform:
            nr_features = len(CONSTANTS.TIME_INVARIANT) + len(CONSTANTS.LAGS)
            X_reshape = np.zeros((X.shape[0], nr_features))
            X_reshape[:, np.arange(len(CONSTANTS.TIME_INVARIANT))] = X[:, CONSTANTS.TIME_INVARIANT]
            X_stack = np.repeat(X_reshape[:, None], len(CONSTANTS.LAGS[0]), axis=1)
            X_stack[:, :, np.arange(len(CONSTANTS.TIME_INVARIANT), nr_features)] = X[:, np.array(CONSTANTS.LAGS).T]

            X = X_stack

        if self.to_fit:
            y = data[:, CONSTANTS.OUTPUT]
            return X, y
        else:
            return X


# define and build a Sequential model
def build_model(model_type='fnn'):
    """

    :param model_type:
    :return:
    """

    model = keras.models.Sequential()

    match model_type:
        case 'fnn':
            nr_features = len(CONSTANTS.TIME_INVARIANT) + sum(len(x) for x in CONSTANTS.LAGS)
            model.add(keras.layers.Dense(512, input_shape=(nr_features,), activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(256, activation='relu'))
            model.add(keras.layers.Dropout(0.1))
            model.add(keras.layers.Dense(128, activation='relu'))
            model.add(keras.layers.Dropout(0.05))
        case 'lstm':
            nr_features = len(CONSTANTS.TIME_INVARIANT) + len(CONSTANTS.LAGS)
            nr_timesteps = len(CONSTANTS.LAGS[0])
            model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(nr_timesteps, nr_features), dropout=0.2))
            model.add(keras.layers.LSTM(8, return_sequences=False, dropout=0.1))

    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model

def get_callbacks(result_dir):

    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [tensorboard_callback,
                 keras.callbacks.ModelCheckpoint(filepath=result_dir + '/model.keras', save_best_only=True),
                 keras.callbacks.CSVLogger(result_dir + '/history.csv')]

    return callbacks

if __name__ == "__main__":

    is_gpu = True    # Detect cpu or gpu
    epochs = 5
    batch_size = 1024
    standardize_file = 'data/meta/std.csv'
    transform = False
    model_type = 'fnn'   # 'fnn', 'lstm'

    # set-up
    if not is_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    basefolder = os.path.basename(__file__).split(".")[0]
    result_dir = f"results/{basefolder}/{datetime.now():%Y-%m-%d_%H_%M_%S}"
    os.makedirs(result_dir)

    print('Initiate data generators \n')
    train_gen = dataloader('data/train1/', batch_size,
                           standardize=True, standardize_file=standardize_file, transform=transform, shuffle=True)
    validate_gen = dataloader('data/validate1/', batch_size,
                              standardize=True, standardize_file=standardize_file, transform=transform)
    test_gen = dataloader('data/test1/', batch_size,
                          standardize=True, standardize_file=standardize_file, transform=transform)

    print('Fit model \n')
    model = build_model(model_type=model_type)
    model.summary() #

    fitted_model = model.fit(
        train_gen,
        validation_data=validate_gen,
        epochs=epochs,
        callbacks=get_callbacks(result_dir),
        use_multiprocessing=True,
        workers=4
    )

    print('Plot training and validation \n')
    plt = pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))
    fig = plt.get_figure()
    fig.savefig(result_dir + "/loss_train_val.png")

    print('Predict on test set \n')
    prediction = model.evaluate(test_gen)
    np.savetxt(result_dir + '/loss_test.txt', prediction, header="test_loss,test_mae", fmt='%1.4f')
