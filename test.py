from utils.load_data import *
from utils.display import *
from models.vae import vae

from keras.models import load_model, model_from_json
import os
import sys
import json

if __name__ == '__main__':

    ########## DIRECTORY SETUP ##########

    ROOT_DIR = "data"
    #TEST_DIR = os.path.join(ROOT_DIR, "test")
    TEST_DIR = os.path.join(ROOT_DIR, "train")
    WEIGHT_DIR = os.path.join("models", "weights")
    model_path = os.path.join(WEIGHT_DIR, "vae.json")

    ########## LOAD DATA ##########

    filenames = [os.path.join(TEST_DIR, x) for x in os.listdir(TEST_DIR)]
    filenames.sort()

    X, filenames = load_images(filenames[:10])
    dims = X.shape[1:]
    print(dims)

    ########## MODEL SETUP ##########

    weight_path = os.path.join(WEIGHT_DIR, "vae_best_weights.hdf5")

    with open(model_path) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(weight_path)

    ########## TEST ##########

    preds = model.predict(X, batch_size=1, verbose=1)
    #preds = [x.astype(np.uint8) for x in preds]

    NUM_TO_VISUALIZE = 5

    affine = nib.load(filenames[0]).affine

    print(X[0,:,:,:].shape, preds[0].shape)

    filename = "test.nii.gz"
    for i in range(NUM_TO_VISUALIZE):
        nii_obj = nib.Nifti1Image(preds[i], affine=affine)
        nib.save(nii_obj, str(i).zfill(3) + "_" + filename)
