from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D,\
    Flatten, Lambda, Reshape, GlobalMaxPooling3D, Conv3DTranspose,\
    AveragePooling3D
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,\
    Flatten, Lambda, Reshape, GlobalMaxPooling2D, Conv2DTranspose,\
    AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import metrics

import matplotlib.pyplot as plt
import numpy as np
import json


def plot_latent_sampling(models,
                         dims,
                         batch_size=128):

    encoder, decoder = models

    # display a 30x30 2D manifold of brains
    n = 30
    brain_size = dims[0] 
    figure = np.zeros((brain_size * n, brain_size * n))
    figure_name_space = "latent_space_sampling.png"
    # linearly spaced coordinates corresponding to the 2D plot
    # of brain classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            brain = x_decoded[0].reshape(brain_size, brain_size)
            figure[i * brain_size: (i + 1) * brain_size,
                   j * brain_size: (j + 1) * brain_size] = brain.T

    plt.figure(figsize=(10, 10))
    start_range = brain_size // 2
    end_range = n * brain_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, brain_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(figure_name_space)
    plt.show()


def vae(model_path, num_channels, dims, learning_rate):

    intermediate_dim = 64
    latent_dim = 2
    epsilon_std = 1.0

    # full vae
    input_img = Input(shape=dims)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    x = Dense(intermediate_dim, activation='relu')(z)
    x = Dense(3*3*4*64, activation='relu')(x)
    x = Reshape(target_shape=(3, 3, 4, 64))(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(num_channels, (3, 3, 3),
                     activation='linear', padding='same')(x)

    vae = Model(input_img, decoded)

    def mae_loss(y_true, y_pred):
        return metrics.mean_absolute_error(K.flatten(y_true),  K.flatten(y_pred))

    def kl_loss(y_true, y_pred):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def disentangled_kl_loss(y_true, y_pred):
        beta = 0.1
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return beta * kl_loss

    def vae_loss(y_true, y_pred):
        return K.mean(mae_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))

    vae.compile(optimizer=Adam(lr=learning_rate),
                loss=vae_loss,
                metrics=[mae_loss, kl_loss, disentangled_kl_loss],)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return vae


def res_vae(model_path, num_channels, dims, ds, learning_rate):

    intermediate_dim = 1024
    latent_dim = 2
    epsilon_std = 1.0

    # full vae
    input_img = Input(shape=dims)

    x = Conv3D(64, (3, 3), strides=(2, 2),
               activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    y = Activation('relu')(x)
    x = Conv3D(64, (3, 3), strides=(1, 1),
               activation='relu', padding='same')(y)
    x = BatchNormalization()(x)
    x = add([x, y])
    x = Activation('relu')(x)

    for _ in range(7):
        x = Conv3D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = Conv3D(32, (3, 3), strides=(1, 1), padding='same')(y)
        x = BatchNormalization()(x)
        x = add([x, y])
        x = Activation('relu')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    x = Dense(intermediate_dim, activation='relu')(z)
    x = Dense(2 * 2 * 32, activation='relu')(x)
    x = Reshape(target_shape=(2, 2, 32))(x)
    x = UpSampling3D((2, 2))(x)

    for _ in range(7):
        x = Conv3D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = Conv3D(32, (3, 3), strides=(1, 1), padding='same')(y)
        x = BatchNormalization()(x)
        x = add([x, y])
        x = Activation('relu')(x)
        x = UpSampling3D((2, 2))(x)

    x = Conv3D(num_channels, (3, 3), activation='linear', padding='same')(x)

    # here is a second stage decode chance
    for _ in range(2):
        x = inception_module(x, ds=ds)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = inception_module(x, ds=ds)
        x = BatchNormalization()(x)
        x = add([x, y])

    decoded = Conv3D(num_channels, (3, 3),
                     activation='linear', padding='same')(x)

    vae = Model(input_img, decoded)

    def vae_loss(y_true, y_pred):
        xent_loss = metrics.mean_squared_error(
            K.flatten(y_true),
            K.flatten(y_pred))
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    vae.compile(optimizer=Adam(lr=learning_rate), loss=vae_loss)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return vae


def inception_module(prev_layer, ds=2):
    a = Conv3D(64//ds, (1, 1, 1), strides=(1, 1, 1),
               padding='same')(prev_layer)

    b = Conv3D(96//ds, (1, 1, 1), strides=(1, 1, 1),
               activation='relu', padding='same')(prev_layer)
    b = Conv3D(128//ds, (3, 3, 3), strides=(1, 1, 1), padding='same')(b)

    c = Conv3D(96//ds, (1, 1, 1), strides=(1, 1, 1),
               activation='relu', padding='same')(prev_layer)
    c = Conv3D(128//ds, (5, 5, 5), strides=(1, 1, 1), padding='same')(c)

    d = AveragePooling3D((3, 3, 3), strides=(
        1, 1, 1), padding='same')(prev_layer)
    d = Conv3D(32//ds, (1, 1, 1), strides=(1, 1, 1), padding='same')(d)

    e = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(prev_layer)
    e = Conv3D(32//ds, (1, 1, 1), strides=(1, 1, 1), padding='same')(e)

    out_layer = concatenate([a, b, c, d, e], axis=-1)

    return out_layer


def inception_vae(model_path, num_channels, dims, ds, learning_rate):

    intermediate_dim = 16
    latent_dim = 2
    epsilon_std = 1.0

    # full vae
    input_img = Input(shape=dims)

    x = inception_module(input_img, ds=ds)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    # residual inception modules
    for _ in range(4):
        x = inception_module(x, ds=ds)
        y = Activation('relu')(x)
        x = inception_module(x, ds=ds)
        x = add([x, y])
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    x = Dense(intermediate_dim, activation='relu')(z)
    x = Dense(3*3*4*(384//ds), activation='relu')(x)
    x = Reshape(target_shape=(3, 3, 4, (384//ds)))(x)

    for _ in range(5):
        x = UpSampling3D((2, 2, 2))(x)
        x = inception_module(x, ds=ds)
        y = Activation('relu')(x)
        x = inception_module(x, ds=ds)
        x = add([x, y])
        x = Activation('relu')(x)

    decoded = Conv3D(num_channels, (3, 3, 3),
                     activation='linear', padding='same')(x)

    vae = Model(input_img, decoded)

    def mae_loss(y_true, y_pred):
        return metrics.mean_absolute_error(K.flatten(y_true),  K.flatten(y_pred))

    def kl_loss(y_true, y_pred):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def disentangled_kl_loss(y_true, y_pred):
        beta = 5
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return beta * kl_loss

    def vae_loss(y_true, y_pred):
        return K.mean(mae_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))

    vae.compile(optimizer=Adam(lr=learning_rate),
                loss=vae_loss,
                metrics=[mae_loss, kl_loss, disentangled_kl_loss],)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return vae


def inception_module_2D(prev_layer, ds=2):
    a = Conv2D(64//ds, (1, 1), strides=(1, 1), padding='same')(prev_layer)

    b = Conv2D(96//ds, (1, 1), strides=(1, 1),
               activation='relu', padding='same')(prev_layer)
    b = Conv2D(128//ds, (3, 3), strides=(1, 1), padding='same')(b)

    c = Conv2D(96//ds, (1, 1), strides=(1, 1),
               activation='relu', padding='same')(prev_layer)
    c = Conv2D(128//ds, (5, 5), strides=(1, 1), padding='same')(c)

    d = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(prev_layer)
    d = Conv2D(32//ds, (1, 1), strides=(1, 1), padding='same')(d)

    e = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(prev_layer)
    e = Conv2D(32//ds, (1, 1), strides=(1, 1), padding='same')(e)

    out_layer = concatenate([a, b, c, d, e], axis=-1)

    return out_layer


def inception_vae_2D(model_path, num_channels, dims, ds, learning_rate):

    intermediate_dim = 2
    latent_dim = 2
    epsilon_std = 1.0

    # full vae
    input_img = Input(shape=dims)

    x = inception_module_2D(input_img, ds=ds)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # residual inception modules
    for _ in range(4):
        x = inception_module_2D(x, ds=ds)
        y = Activation('relu')(x)
        x = inception_module_2D(x, ds=ds)
        x = add([x, y])
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    x = Dense(intermediate_dim, activation='relu')(z)
    x = Dense(3*3*(384//ds), activation='relu')(x)
    x = Reshape(target_shape=(3, 3, (384//ds)))(x)

    for _ in range(5):
        x = UpSampling2D((2, 2))(x)
        x = inception_module_2D(x, ds=ds)
        y = Activation('relu')(x)
        x = inception_module_2D(x, ds=ds)
        x = add([x, y])
        x = Activation('relu')(x)

    decoded = Conv2D(num_channels, (3, 3),
                     activation='linear', padding='same')(x)

    vae = Model(input_img, decoded)

    def mae_loss(y_true, y_pred, dims=dims):
        # need to scale up by number of pixels/voxels
        scaling = np.prod(dims)
        return scaling * metrics.mean_absolute_error(K.flatten(y_true),  K.flatten(y_pred))

    def kl_loss(y_true, y_pred):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def disentangled_kl_loss(y_true, y_pred):
        beta = 1  # 1e-7
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return beta * kl_loss

    def vae_loss(y_true, y_pred):
        return K.mean(mae_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))

    vae.compile(optimizer=Adam(lr=learning_rate),
                loss=vae_loss,
                metrics=[mae_loss, kl_loss, disentangled_kl_loss],)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return vae


def vae_2D(model_path, num_channels, dims, ds, learning_rate):

    intermediate_dim = 2
    latent_dim = 2
    epsilon_std = 1.0

    ########## ENCODER ##########
    input_img = Input(shape=dims)
    x = Conv2D(64//ds, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    for _ in range(4):
        x = Conv2D(64//ds, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    ########## END ENCODER ##########

    ########## DECODER ##########
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = Dense(3*3*(64//ds), activation='relu')(x)
    x = Reshape(target_shape=(3, 3, (64//ds)))(x)

    for _ in range(5):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64//ds, (3, 3), activation='relu', padding='same')(x)

    decoded = Conv2D(num_channels, (3, 3),
                     activation='linear', padding='same')(x)
    ########## END DECODER ##########

    # instantiate encoder/decoder
    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, decoded, name='decoder')

    # here, the [2] index selects the 'z' tensor output as input to decoder
    # after passing inputs through the encoder
    output = decoder(encoder(input_img)[2])
    vae = Model(input_img, output)

    def mae_loss(y_true, y_pred, dims=dims):
        # need to scale up by number of pixels/voxels
        scaling = np.prod(dims)
        return scaling * metrics.mean_absolute_error(K.flatten(y_true),  K.flatten(y_pred))

    def kl_loss(y_true, y_pred):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def disentangled_kl_loss(y_true, y_pred):
        beta = 1e-7
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return beta * kl_loss

    def vae_loss(y_true, y_pred):
        return K.mean(mae_loss(y_true, y_pred) + disentangled_kl_loss(y_true, y_pred))

    vae.compile(optimizer=Adam(lr=learning_rate),
                loss=vae_loss,
                metrics=[mae_loss, kl_loss, disentangled_kl_loss],)

    print(vae.summary())

    json_string = vae.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    return encoder, decoder, vae
