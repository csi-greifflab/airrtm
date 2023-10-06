import numpy as np
import tensorflow as tf


@tf.keras.saving.register_keras_serializable('airrtm')
class ReconstructionLoss(tf.keras.losses.Loss):
    def __init__(self, coef, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef

    def call(self, y_true, y_pred, **kwargs):
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, **kwargs)
        )
        return reconstruction_loss * self.coef


@tf.keras.saving.register_keras_serializable('airrtm')
class BCLossCoef(tf.keras.losses.Loss):
    def __init__(self, coef, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        self.bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred, **kwargs):
        return self.bc_loss(y_true, y_pred, **kwargs) * self.coef


@tf.keras.saving.register_keras_serializable('airrtm')
class KLDLossCoef(tf.keras.losses.Loss):
    def __init__(self, coef, **kwargs):
        super().__init__(**kwargs)
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.coef = coef

    def call(self, y_true, y_pred, **kwargs):
        return self.mae_loss(y_true, y_pred, **kwargs) * self.coef


@tf.keras.saving.register_keras_serializable('airrtm')
class BCLossScaled(tf.keras.metrics.BinaryCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return (super().result() + np.log(0.5)) * 1e3


@tf.keras.saving.register_keras_serializable('airrtm')
class KLDMetric(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, coef, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return super().result() * self.coef

