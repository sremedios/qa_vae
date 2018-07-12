from utils.load_data import *
from utils.display import *
from models.vae import vae, inception_vae

from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
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

    filenames = [os.path.join(TRAIN_DIR, x) for x in os.listdir(TRAIN_DIR)]
    filenames.sort()

    X, filenames = load_images(filenames)
    dims = X.shape[1:]
    print(dims)

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
    model = inception_vae(model_path=model_path,
                          num_channels=X.shape[-1],
                          ds=8,
                          dims=dims,
                          learning_rate=1e-4)

    '''
    model = vae(model_path=model_path,
                num_channels=X.shape[-1],
                dims=dims,
                learning_rate=1e-4)
    '''

    ########## TRAIN ##########

    model.fit(X, X,
              validation_split=0.2,
              epochs=10000000,
              batch_size=1,
              callbacks=callbacks_list)
