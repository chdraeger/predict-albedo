from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import pyreadr


# def generator(path, batch_size):
#     print('start generator')
#     files = glob.glob(path + '*' + '.RDS')
#     for file in files:
#         print('file', file)
#         try:
#             df = pyreadr.read_r(file)[None].to_numpy()
#             batches = int(np.ceil(len(df)/batch_size))
#             for i in range(batches):
#                 df_chunk = df[i*batch_size:min(len(df), i*batch_size+batch_size)]
#                 x = df_chunk[:, :-1]
#                 y = df_chunk[:, -1]
#                 yield x, y
#
#         except EOFError:
#             print("error" + file)


# prepare input data
class dataloader(tf.keras.utils.Sequence):
    def __init__(self,path,batch_size,shuffle=None,transform=None):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.filenames = self.get_all_files_in_path(path)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size))) # total number of batches in the full dataset

    def get_all_files_in_path(self,path):
        # filenames =
        # np.random.shuffle(train_file_names)
        return glob.glob(path + '*' + '.RDS')

    def __getitem__(self, index): # length of sequence
        file = self.filenames[index]
        data = pyreadr.read_r(file)[None].to_numpy()
        x = data[:, :-1]
        y = data[:, -1]
        return x, y



def create_dataset():
    list_train = ['data/train/' + str(s) for s in listdir('data/train/')]

    data_tr = pyreadr.read_r(list_train[0])[None].to_numpy()

    # standardize
    mean = np.loadtxt("data/meta/mean_all.txt", dtype=float)
    std = np.loadtxt("data/meta/std_all.txt", dtype=float)
    data_tr_std = (data_tr - mean) / std
    x = data_tr_std[:, :-1]
    y = data_tr_std[:, -1]
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

# fit the model
def fit_model(epochs, batch_size, validation_split):
    x_train, y_train = create_dataset()
    model = build_model()
    build_model().summary()

    callbacks = [ModelCheckpoint(filepath='model.hdf5',
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min'),
        CSVLogger('history.csv')
    ]

    fitted_model = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks
    )

    # plot
    plt = pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))
    fig = plt.get_figure()
    fig.savefig("model.png")


if __name__ == "__main__":
    # specifications
    epochs = 2 # 30
    batch_size = 40000 # 2**7
    path_train = 'data/test1/'

    # get data
    train_generator = dataloader(path_train, batch_size)
    #transform = None
    for i, data in enumerate(train_generator):
        print(i)

    # fit_model(epochs, batch_size, validation_split)
    model = build_model()
    build_model().summary()

    fitted_model = model.fit(
        train_generator,
        validation_data=train_generator,
        epochs=epochs
    )

    print('here')

    # dataloader_train = christinas_dataloader(path_train,transform,batch_size)
    # for i in range(len(dataloader_train)):
    #     data_file = dataloader_train.__getitem__(i)
    #
    # for data_file in dataloader_train:
    #     niter = len(data_file)/dataloader_train.batch_size
    #     for i in range(niter):
    #         data = data_file[i*batch_size:(i+1)*batch_size]
    #         input_data, output_data = data
    #         output_prediction = model(input_data)
    #         loss = loss_fn(output_data,output_prediction)
    #         model.backward()

    # dataloader_train.get_all_files_in_path()
    # dataloader_train.__len__()
    # len(dataloader_train)
