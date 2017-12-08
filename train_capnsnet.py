def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf
from keras.utils import to_categorical
from capsule.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.datasets import mnist
from keras.utils import multi_gpu_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))
    
input_shape=x_train.shape[1:]
n_class=len(np.unique(np.argmax(y_train, 1)))
num_routing=3
lam_recon=0.392
lr=0.001
print(input_shape)

with tf.device('/gpu:0'):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)
    print(digitcaps.shape)
    # # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)
    print(out_caps.shape)
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    
train_model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., lam_recon]
                  metrics={'capsnet': 'accuracy'})