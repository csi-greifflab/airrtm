import os
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_probability as tfp

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import alphabet_size
from losses import ReconstructionLoss, BCLossCoef, KLDLossCoef, BCLossScaled, KLDMetric




@tf.keras.saving.register_keras_serializable('airrtm')
class AIRRTM(tf.keras.Model):
    # @staticmethod
    # def correlation_reg(weight_matrix, n_items=None, transpose=False):
    #     if transpose:
    #         weight_matrix = tf.transpose(weight_matrix)
    #     if n_items is None:
    #         n_items = weight_matrix.shape[0]
    #     corr_matrix_sq = tfp.stats.correlation(
    #         weight_matrix,
    #         weight_matrix,
    #     ) ** 2
    #     total_correlation = tf.reduce_sum(corr_matrix_sq) - n_items
    #     mean_correlation = total_correlation / (n_items * (n_items - 1))
    #     return mean_correlation

    # @staticmethod
    # def entropy_reg(weight_matrix, n_items=None, transpose=False):
    #     if len(weight_matrix.shape) != 2:
    #         raise Exception
    #     if transpose:
    #         weight_matrix = tf.transpose(weight_matrix)
    #     if n_items is None:
    #         n_items = weight_matrix.shape[1]
    #     probs_matrix = tf.keras.activations.softmax(weight_matrix, axis=0)
    #     entropy = tf.reduce_sum(- probs_matrix * tf.math.log(probs_matrix)) / n_items
    #     return entropy


    # def topic_proportions_reg(weight_matrix, entropy_coef, correlation_coef, l2_coef):
    #     return 0 + \
    #         tf.reduce_mean(weight_matrix**2)**0.5 * l2_coef + \
    #         AIRRTM.entropy_reg(weight_matrix) * entropy_coef
    #         # AIRRTM.correlation_reg(weight_matrix) * 0 + \    

    #latent vector sampling
    @staticmethod
    def sampling(args):
        z_mean, z_log_sigma = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

    class SamplingLayer(tf.keras.layers.Layer):
        def __init__(self, dim, *args, **kwargs):
            super().__init__()
            self.dim = dim
            sigma_init = tf.random_normal_initializer()
            self.sigma = self.add_weight(
                name="sigma",
                initializer="random_normal",
                trainable=True
            )

        def get_config(self):
            config = super().get_config()
            config.update({
                "dim": self.dim,
            })
            return config

        def call(self, inputs):
            z_mean = inputs
            z_log_sigma = tf.ones(self.dim) * self.sigma
            return AIRRTM.sampling([z_mean, z_log_sigma]), z_log_sigma    

    def __init__(
        self,
        sample_labels,
        max_len=20,
        alphabet_size=alphabet_size,
        n_topics_signal=40,
        n_topics_nonsignal=40,
        tm_signal_coef=0.5,
        latent_dim=40,
        n_samples=100,
        decorrelation_regularizer_coef=0.0,
        entropy_regularizer_coef=0.0,
        topic_proportions_l2_coef=1e-2,
        kl_coef=1.0,
        tm_likelihood_coef=1.0,
        label_likelihood_coef=1.0,
        reconstruction_loss_coef=4e2,
        latent_space_to_topic_proportions_coef=1.0,
    ):
        super().__init__()
        self.optimizer = None
        self.sample_labels = np.array(sample_labels)
        self.max_len = max_len
        self.alphabet_size = alphabet_size
        self.n_topics_signal = n_topics_signal
        self.n_topics_nonsignal = n_topics_nonsignal
        self.tm_signal_coef = tm_signal_coef
        self.n_topics = self.n_topics_signal + self.n_topics_nonsignal
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.decorrelation_regularizer_coef = decorrelation_regularizer_coef
        self.entropy_regularizer_coef = entropy_regularizer_coef
        self.topic_proportions_l2_coef = topic_proportions_l2_coef
        self.kl_coef = kl_coef
        self.tm_likelihood_coef = tm_likelihood_coef
        self.label_likelihood_coef = label_likelihood_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.latent_space_to_topic_proportions_coef = latent_space_to_topic_proportions_coef

        self.seq_input_layer = tf.keras.layers.Input(shape=(self.max_len, 1), name="sequence_input")
        self.seq_one_hot_encoding_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.CategoryEncoding(
                num_tokens=self.alphabet_size,
                output_mode='one_hot',
            ),
            name='seq_one_hot_encoder',
        )
        self.sample_input_layer = tf.keras.layers.Input(shape=(1,), name="repertoire_input")

        self.sample_topic_proportions_layer = tf.keras.layers.Embedding(
            input_dim=self.n_samples,
            output_dim=self.n_topics,
            # embeddings_regularizer=lambda M: AIRRTM.topic_proportions_reg(
            #     M,
            #     self.entropy_regularizer_coef,
            #     0,
            #     self.topic_proportions_l2_coef
            # ),
            name="topic_proportions"
        )
        self.topic_proportions_reshape_layer = tf.keras.layers.Reshape(target_shape=(self.n_topics,))
        # self.encode_layer = tf.keras.models.Sequential(
        #     [
        #         tf.keras.layers.Conv1D(filters=self.n_topics, kernel_size=max_len//2),
        #         tf.keras.layers.PReLU(),
        #         tf.keras.layers.GlobalMaxPool1D(name="EncoderMaxPool_global"),
        #     ],
        #     name='encoder_conv',
        # )
        self.encode_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.n_topics, return_sequences=False, return_state=False
            ),
            name="encoder_1"
        )
        self.encode_layer_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.latent_dim, return_sequences=False, return_state=False
            ),
            name="encoder_2"
        )
        # self.encode_layer_2 = tf.keras.layers.LSTM(units=self.latent_dim, return_sequences=False, return_state=False, name="encoderLSTM")
        self.concat_layer = tf.keras.layers.Concatenate()
        self.sampling_layer = AIRRTM.SamplingLayer(self.latent_dim)
        self.pre_z_mean_layer_act = tf.keras.layers.PReLU()
        self.z_mean_layer = tf.keras.layers.Dense(
            self.latent_dim,
            name='z_mean',
            kernel_initializer=tf.keras.initializers.Orthogonal(),
        )

        self.latent_space_to_topic_proportions_layer = tf.keras.layers.Dense(
            self.n_topics,
            kernel_initializer=tf.keras.initializers.Orthogonal(),
            name='latent_to_topics',
            # activation='softmax',
            activity_regularizer=tf.keras.regularizers.L2(l2=self.latent_space_to_topic_proportions_coef),

        )
        
        self.z_log_sigma_layer = tf.keras.layers.Dense(self.n_topics,name='z_sigma')
        
        self.dot_product_layer = tf.keras.layers.Dot(axes=1, name="tm_likelihood")

        self.decode_repeating_layer = tf.keras.layers.RepeatVector(self.max_len, name='repeat_latent_vector')
        self.decode_lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.latent_dim, return_sequences=True),
            name='decoder_lstm'
        )

        self.decode_dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.alphabet_size, name='DecoderDense'))
        self.decode_softmax_layer = tf.keras.layers.Softmax(name="decoded_sequence") 

        self.kl_divergence_layer = tf.keras.layers.Layer(name="kl_divergence")
        self.correlation_reg_output_layer = tf.keras.layers.Layer(name="correlation_reg")
        self.label_prediction_layer = tf.keras.layers.Dense(1, name='label_likelihood', activation='sigmoid')
        
    def call(self, inputs):
        # forward
        topic_proportions = self.sample_topic_proportions_layer(
            inputs['repertoire_input']
        )
        # topic_proportions = tf.keras.activations.softmax(topic_proportions)
        topic_proportions = self.topic_proportions_reshape_layer(topic_proportions)

        one_hot_seq = self.seq_one_hot_encoding_layer(
            inputs['sequence_input']
        )
        encoded_seq_1 = self.encode_layer(one_hot_seq)
        encoded_seq_2 = self.encode_layer_2(one_hot_seq)
        encoded_seq = self.concat_layer([encoded_seq_1, encoded_seq_2])
        z_mean = self.pre_z_mean_layer_act(encoded_seq)
        z_mean = self.z_mean_layer(z_mean)
        z, z_log_sigma = self.sampling_layer(z_mean)

        seq_topic_probabilities = self.latent_space_to_topic_proportions_layer(z)

        # seq_total_probability_output = self.dot_product_layer([topic_proportions, seq_topic_probabilities])
        seq_total_probability_signal = tf.keras.activations.sigmoid(self.dot_product_layer([
            topic_proportions[:, :self.n_topics_signal],
            seq_topic_probabilities[:, :self.n_topics_signal],
        ]))
        seq_total_probability_nonsignal = tf.keras.activations.sigmoid(self.dot_product_layer([
            topic_proportions[:, self.n_topics_signal:],
            seq_topic_probabilities[:, self.n_topics_signal:],
        ]))
        seq_total_probability_output = self.tm_signal_coef * seq_total_probability_signal + (1 - self.tm_signal_coef) * seq_total_probability_nonsignal
        label_output = self.label_prediction_layer(topic_proportions[:, :self.n_topics_signal]) # seq_topic_probabilities
        # label_output = self.label_prediction_layer(seq_topic_probabilities[:, :self.n_topics_signal]) # seq_topic_probabilities

        kl_divergence = - 0.5 * tf.reduce_sum(1 + z_log_sigma - tf.math.square(z_mean) - tf.math.exp(z_log_sigma), axis=-1)
        kl_divergence_output = self.kl_divergence_layer(kl_divergence)
        
        decoded_sequence = self.decode_repeating_layer(z)
        decoded_sequence = self.decode_lstm_layer(decoded_sequence)
        decoded_sequence = self.decode_dense_layer(decoded_sequence)
        decoded_sequence_output = self.decode_softmax_layer(decoded_sequence)
        return {
            "tm_likelihood": seq_total_probability_output,
            "label_likelihood": label_output,
            "decoded_sequence": decoded_sequence_output,
            "kl_divergence": kl_divergence_output,
        }        

    def compile_model(self, lr=1e-4):
        bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        tm_loss = BCLossCoef(coef=self.tm_likelihood_coef)
        label_loss = BCLossCoef(coef=self.label_likelihood_coef)
        reconstruction_loss = ReconstructionLoss(coef=self.reconstruction_loss_coef)
        kld_loss = KLDLossCoef(coef=self.kl_coef)

        tm_loss_scaled = BCLossScaled()
        label_loss_scaled = BCLossScaled()
        kld_pure = KLDMetric(coef=1 / self.latent_dim)
        
        if self.optimizer is None:
            logging.info(f'Compiling for the first time – creating a fresh optimizer...')
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.compile(
            loss={
                "tm_likelihood": tm_loss,
                "label_likelihood": label_loss,
                "decoded_sequence": reconstruction_loss,
                "kl_divergence": kld_loss,
            },
            optimizer=self.optimizer,
            metrics={
                "tm_likelihood": ["acc", tm_loss_scaled],
                "label_likelihood": [],
                "decoded_sequence": ["acc"],
                "kl_divergence": [],
            },
            weighted_metrics={
                "tm_likelihood": [],
                "label_likelihood": ["acc", label_loss_scaled],
                "decoded_sequence": [],
                "kl_divergence": [kld_pure],
            }
        )

    def fit_model(self, batch_size, epochs, input_data, callbacks=[]):
        dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes = input_data
        self.fit(
            x={
                "sequence_input": dataset_seq,
                "repertoire_input": dataset_repertoire_id
            },
            y={
                "tm_likelihood": dataset_tm_target,
                "label_likelihood": dataset_repertoire_label,
                "decoded_sequence": dataset_seq,
                "kl_divergence": dataset_kl_target,
                # "correlation_reg": dataset_kl_target,
            },
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[]+callbacks,
            sample_weight=sample_weights.reshape(-1, 1),
        )
    
    def get_topic_proportion_matrix(self):
        return self.sample_topic_proportions_layer.get_weights()[0]

    def encode_sequences(self, seqs):
        one_hot_seqs = self.seq_one_hot_encoding_layer(seqs)
        encoded_seq_1 = self.encode_layer(one_hot_seqs)
        encoded_seq_2 = self.encode_layer_2(one_hot_seqs)
        encoded_seq = tf.concat([encoded_seq_1, encoded_seq_2], axis=-1)
        z_mean = self.pre_z_mean_layer_act(encoded_seq)
        z_mean = self.z_mean_layer(z_mean)
        return z_mean

    def decode_sequences(self, encoded_seqs):
        decoded_sequence = self.decode_repeating_layer(encoded_seqs)
        decoded_sequence = self.decode_lstm_layer(decoded_sequence)
        decoded_sequence = self.decode_dense_layer(decoded_sequence)
        return decoded_sequence

    def predict_topic_probs(self, seqs):
        return self.latent_space_to_topic_proportions_layer(
            self.encode_sequences(seqs)
        ).numpy()

    def predict_signal_intensity(self, seqs):
        sample_topic_p = tf.keras.activations.softmax(self.sample_topic_proportions_layer.weights[0]).numpy()
        sample_topic_p_df = pd.DataFrame(sample_topic_p[:, :self.n_topics_signal])
        topic_diffs = sample_topic_p_df.loc[self.sample_labels == 1].mean(axis=0) - sample_topic_p_df.iloc[self.sample_labels == 0].mean(axis=0)
        best_topic_id = topic_diffs.argmax()
        topic_diffs_weights = topic_diffs.abs() / topic_diffs.abs().sum() * topic_diffs / topic_diffs.abs()
        predicted_topics = self.predict_topic_probs(seqs)[:, :self.n_topics_signal]
        signal_intensity = (predicted_topics  * topic_diffs_weights.to_numpy()).sum(axis=1)
        return signal_intensity

    def predict_signal_intensity_2(self, seqs):
        predicted_topics = self.predict_topic_probs(seqs)
        signal_intensity = self.label_prediction_layer(predicted_topics[:, :self.n_topics_signal]).numpy().flatten()
        return signal_intensity

def seq_tensor_to_seq(seq_tensor):
    alphabet_encoder = LabelEncoder()
    alphabet_encoder.fit(alphabet)
    m = np.argmax(seq_tensor.numpy(), axis=-1)
    sequences = [''.join(alphabet_encoder.inverse_transform(s)) for s in m]
    return sequences


def get_default_model(max_len, sample_labels):
    n_samples = len(sample_labels)

    tm_coef = 0.5  # tm_coef = 0.995
    vae_coef = 1 - tm_coef

    tm_likelihood_coef=0.7
    label_likelihood_coef=(1.0 - tm_likelihood_coef)

    reconstruction_loss_coef=0.75  # reconstruction_loss_coef=0.95
    kl_coef=(1.0 - reconstruction_loss_coef)

    decorrelation_regularizer_coef = 0.01
    latent_space_to_topic_proportions_coef = 0.001

    n_topics_signal = 3  # n_topics_signal = 4
    n_topics_nonsignal = 4  # n_topics_nonsignal = 4
    tm_signal_coef = 0.4
    n_topics = n_topics_signal + n_topics_nonsignal
    latent_dim = 70
    airrtm_model = AIRRTM(
        sample_labels=sample_labels,
        max_len=max_len,
        n_samples=n_samples,
        n_topics_signal=n_topics_signal,
        n_topics_nonsignal=n_topics_nonsignal,
        tm_signal_coef=tm_signal_coef,
        latent_dim=latent_dim,
        decorrelation_regularizer_coef=decorrelation_regularizer_coef,
        entropy_regularizer_coef=0.0,
        topic_proportions_l2_coef=1e-4,
        tm_likelihood_coef=tm_coef * tm_likelihood_coef,
        label_likelihood_coef=tm_coef * label_likelihood_coef,
        reconstruction_loss_coef=vae_coef * reconstruction_loss_coef,
        kl_coef=vae_coef * kl_coef / latent_dim,
        latent_space_to_topic_proportions_coef=latent_space_to_topic_proportions_coef,
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=50000,
        decay_rate=0.5
    )
    airrtm_model.compile_model(lr=lr_schedule)
    return airrtm_model


def load_model(model_file):
    airrtm_model = tf.keras.models.load_model(model_file, compile=False)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=50000,
        decay_rate=0.5
    )
    airrtm_model.compile_model(lr=lr_schedule)
    return airrtm_model
