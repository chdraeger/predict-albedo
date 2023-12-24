#from tensorflow.keras.models import Sequential
#from tensorflow.keras import layers
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.callbacks import CSVLogger
from os import listdir
import pandas as pd
import numpy as np
import pyreadr


def generator(path):
    print('start generator')
    files = [path + str(s) for s in listdir(path)]
    while True:
        print('loop generator')
        for file in files:
            try:
                df = pyreadr.read_r(file)[None].to_numpy() #test
                batches = int(np.ceil(len(df)/batch_size))
                for i in range(batches):
                    yield df[i*batch_size:min(len(df), i*batch_size+batch_size)]

            except EOFError:
                print("error" + file)


# prepare input data
class christinas_dataloader:
    def __init__(self,path,transform,batch_size):
        self.path = path
        self.transform = transform
        self.batch_size = batch_size
        #self.get_all_files_in_path(path)

    def get_all_files_in_path(self,path):
        glob(path)
        self.files = files # loc files
        return

    def __getitem__(self, index): # right now: files number
        file = self.files[index]
#        with file open as f:
#            data = np.load(file)
        return data

    def __len__(self):
        return self.n



def create_dataset():

    list_train = ['data/train/' + str(s) for s in listdir('data/train/')]

    data_tr = pyreadr.read_r(list_train[0])[None].to_numpy()

    # standardize
    mean = np.loadtxt("data/meta/mean_all.txt", dtype=float)
    std = np.loadtxt("data/meta/std_all.txt", dtype=float)
    data_tr_std = (data_tr - mean) / std
    data_tr_std[0:20]
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
    epochs = 30
    batch_size = 2**7
    validation_split = 0.1

    path_train = 'data/train/'
    data_generator = generator(path_train)
    for i, data in enumerate(data_generator):
        print(i)

    # fit_model(epochs, batch_size, validation_split)
    #transform = ?
    transform = None

    dataloader_train = christinas_dataloader(path_train,transform,batch_size)
    for i in range(len(dataloader_train)):
        data_file = dataloader_train.__getitem__(i)

    for data_file in dataloader_train:
        niter = len(data_file)/dataloader_train.batch_size
        for i in range(niter):
            data = data_file[i*batch_size:(i+1)*batch_size]
            input_data, output_data = data
            output_prediction = model(input_data)
            loss = loss_fn(output_data,output_prediction)
            model.backward()

    dataloader_train.get_all_files_in_path()
    dataloader_train.__len__()
    len(dataloader_train)
