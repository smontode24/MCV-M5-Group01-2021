from keras.utils import plot_model
from keras.layers import Flatten, Dense
from keras.models import Model
import keras
import numpy as np
from keras import regularizers
from keras.layers import Flatten, Dense, Reshape, Dropout, BatchNormalization, MaxPooling2D, Conv2D, Activation
from functools import partial 
from keras import layers

# Keras model definition

class RandomNet:
    def __init__(self, dropout_prob=0.5, reg_strength=1e-5):
        super().__init__()
        inputs = keras.Input(shape=(256, 256, 3))

        # Block 1
        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), use_bias=False, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        residual = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal", use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 2
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization()(x)

        # Block 2 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        residual = Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer="he_normal", padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # Block 3
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization()(x)

        # Block 3 Pool
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        residual = Conv2D(256, (3, 3), strides=(2, 2), kernel_initializer="he_normal", padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer="he_normal", padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization()(x)

        # Block 13
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer="he_normal", padding='same', use_bias=False)(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization()(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        # Dropout added
        x = Dropout(dropout_prob)(x)
        outputs = Dense(8, activation='softmax', name='predictions', kernel_initializer="glorot_normal")(x)
        
        self.model = Model(inputs, outputs)
        self.model.trainable = True
