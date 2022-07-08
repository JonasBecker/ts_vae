from gc import callbacks
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import pickle
import math
from keras import backend as K
latent_dim = 3
beta = 1


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.r2_loss_tracker = tfa.metrics.RSquare(name='r2_loss')
        self.val_r2_loss_tracker = tfa.metrics.RSquare(name='val_r2_loss')
   

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.r2_loss_tracker
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * beta
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.r2_loss_tracker.update_state(data, reconstruction)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "r_square": self.r2_loss_tracker.result()
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * beta
        total_loss = reconstruction_loss + kl_loss
        self.val_r2_loss_tracker.update_state(data, reconstruction)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "r_square": self.val_r2_loss_tracker.result()
        }
        
    # def call(self, data):
    #     z_mean, z_log_var, z = self.encoder(data)
    #     reconstruction = self.decoder(z)
    #     reconstruction_loss = tf.reduce_mean(
    #         tf.reduce_sum(
    #             keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
    #         )
    #     )
    #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * self.beta
    #     total_loss = reconstruction_loss + kl_loss
        
        
    #     self.add_metric(kl_loss, name="kl_loss", aggregation="mean")
    #     self.add_metric(total_loss, name="total_loss", aggregation="mean")
    #     self.add_metric(
    #         reconstruction_loss, name="reconstruction_loss", aggregation="mean"
    #     )
    #     return reconstruction
    
def remap(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

# class CustomCallback(tf.keras.callbacks.Callback):
#     def __init__(self, cyclic_width, beta_arr):
#         super(CustomCallback, self).__init__()
#         self.iter = 0
#         self.cyclic_width = cyclic_width
#         self.beta_arr = beta_arr
        
#     def on_train_batch_start(self, batch, logs = None):
#         K.set_value(self.model.beta, self.beta_val())

    
#     def on_train_batch_end(self, batch, logs = None):
#         self.iter +=1 
#         if self.iter == self.cyclic_width:
#             self.iter = 0
            
#     def beta_val(self):
#         return self.beta_arr[(len(self.beta_arr)-1) * self.iter//(self.cyclic_width-1)]            
        
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

train_gaf_ds = np.load('data/train_gaf_ds.npy')
val_gaf_ds = np.load('data/val_gaf_ds.npy')
test_gaf_ds = np.load('data/val_gaf_ds.npy')

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
#beta_arr = np.zeros(100)
#cc = CustomCallback(cyclic_width = 150, beta_arr=beta_arr)
hist = vae.fit(train_gaf_ds, epochs=60, batch_size=128, validation_data = (val_gaf_ds, val_gaf_ds), validation_batch_size = 128,  )

with open(f"data/model_history_beta_{beta}.pkl", 'wb') as f:
    pickle.dump(hist.history, f)

vae.encoder.save(f"data/encoder_beta_{beta}")
vae.decoder.save(f"data/decoder_beta_{beta}")


print(len(train_gaf_ds))
split = ['train', 'val', 'test']
gaf_all = [train_gaf_ds, val_gaf_ds, test_gaf_ds]

for i, ds in enumerate(gaf_all):
    z_mean, z_log_var, z = vae.encoder.predict(ds, 128)
    np.save(f'data/z_all_{split[i]}_beta_{beta}', np.stack([z_mean, z_log_var, z],0))
    imgs = vae.decoder.predict(z,128)
    np.save(f'data/dec_imgs_{split[i]}_beta_{beta}', imgs)

