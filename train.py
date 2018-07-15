from utils.load_data import *
from utils.load_slice_data import *
from utils.display import *
from models.vae import *

from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import time
import sys

if __name__ == '__main__':

    ########## DIRECTORY SETUP ##########

    ROOT_DIR = "data"
    TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    WEIGHT_DIR = os.path.join("models", "weights")
    model_path = os.path.join(WEIGHT_DIR, "vae.json")

    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    ########## LOAD DATA ##########

    X, filenames, dims = load_middle_slice_data(TRAIN_DIR)

    if False:
        limit = 10 
        X = X[0:limit]
        filenames = filenames[0:limit]

    ########## CALLBACKS ##########

    weight_path = os.path.join(WEIGHT_DIR, "vae_best_weights.hdf5")

    mc = ModelCheckpoint(weight_path,
                         monitor='loss',
                         verbose=0,
                         save_best_only=True,
                         mode='auto',)
    es = EarlyStopping(monitor='loss',
                       min_delta=1e-8,
                       patience=300)
    callbacks_list = [mc, es]

    ########## MODEL SETUP ##########
    encoder, decoder, model = vae_2D(model_path=model_path,
                                     num_channels=X.shape[-1],
                                     ds=1,
                                     dims=dims,
                                     learning_rate=1e-4)

    '''
    model = vae(model_path=model_path,
                num_channels=X.shape[-1],
                dims=dims,
                learning_rate=1e-4)
    '''

    ########## TRAIN ##########

    batch_size = 16
    start_time = time.time()
    model.fit(X, X,
              #validation_split=0.2,
              epochs=1000000,
              batch_size=batch_size,
              callbacks=callbacks_list,
              verbose=0)

    print("Elapsed time: {:.4f}s".format(time.time() - start_time))

    plot_latent_sampling((encoder, decoder),
                         dims,
                         batch_size=batch_size)
