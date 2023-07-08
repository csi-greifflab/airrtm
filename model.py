import os
n_threads = 8
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import alphabet_size


tf.config.threading.set_inter_op_parallelism_threads(
    n_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    n_threads
)
tf.config.set_soft_device_placement(True)


class AIRRTM:
    @staticmethod
    def correlation_reg(weight_matrix, n_items=None, transpose=False):
        if transpose:
            weight_matrix = tf.transpose(weight_matrix)
        if n_items is None:
            n_items = weight_matrix.shape[0]
        corr_matrix_sq = tfp.stats.correlation(
            weight_matrix,
            weight_matrix,
        ) ** 2
        total_correlation = tf.reduce_sum(corr_matrix_sq) - n_items
        mean_correlation = total_correlation / (n_items * (n_items - 1))
        return mean_correlation

    @staticmethod
    def entropy_reg(weight_matrix, n_items=None, transpose=False):
        if len(weight_matrix.shape) != 2:
            raise Exception
        if transpose:
            weight_matrix = tf.transpose(weight_matrix)
        if n_items is None:
            n_items = weight_matrix.shape[1]
        probs_matrix = tf.keras.activations.softmax(weight_matrix, axis=0)
        entropy = tf.reduce_sum(- probs_matrix * tf.math.log(probs_matrix)) / n_items
        return entropy

    @staticmethod
    def topic_proportions_reg(weight_matrix, entropy_coef, correlation_coef, l2_coef):
        return 0 + \
            tf.reduce_mean(weight_matrix**2)**0.5 * l2_coef + \
            AIRRTM.entropy_reg(weight_matrix) * entropy_coef
            # AIRRTM.correlation_reg(weight_matrix) * 0 + \

    @staticmethod
    def reconstruction_loss(input_seqs, decoded_seqs):
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(input_seqs, decoded_seqs))
        return reconstruction_loss

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
            super(AIRRTM.SamplingLayer, self).__init__()
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
        max_len=20,
        alphabet_size=alphabet_size,
        n_topics_signal=40,
        n_topics_nonsignal=40,
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
        self.max_len = max_len
        self.alphabet_size = alphabet_size
        self.n_topics_signal = n_topics_signal
        self.n_topics_nonsignal = n_topics_nonsignal
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

        self.seq_input_layer = tf.keras.layers.Input(shape=(self.max_len, self.alphabet_size), name="sequence_input")
        self.sample_input_layer = tf.keras.layers.Input(shape=(1,), name="repertoire_input")

        self.sample_topic_proportions_layer = tf.keras.layers.Embedding(
            input_dim=self.n_samples,
            output_dim=self.n_topics,
            embeddings_regularizer=lambda M: AIRRTM.topic_proportions_reg(
                M,
                self.entropy_regularizer_coef,
                0,
                self.topic_proportions_l2_coef
            ),
            name="topic_proportions"
        )
        self.topic_proportions_reshape_layer = tf.keras.layers.Reshape(target_shape=(self.n_topics,))
        self.encode_layer = tf.keras.models.Sequential([
           tf.keras.layers.Conv1D(filters=self.n_topics, kernel_size=max_len//2, name="EncoderConv_1"),
           tf.keras.layers.PReLU(),
           tf.keras.layers.GlobalMaxPool1D(name="EncoderMaxPool_global"),
        ]) 
        self.encode_layer_2 = tf.keras.layers.LSTM(units=self.latent_dim, return_sequences=False, return_state=False, name="encoderLSTM")
        self.concat_layer = tf.keras.layers.Concatenate()
        self.sampling_layer = AIRRTM.SamplingLayer(self.latent_dim)
        self.pre_z_mean_layer_act = tf.keras.layers.PReLU()
        self.z_mean_layer = tf.keras.layers.Dense(
            self.latent_dim,
            name='MeanVector',
            kernel_initializer=tf.keras.initializers.Orthogonal(),
        )

        self.latent_space_to_topic_proportions_layer = tf.keras.layers.Dense(
            self.n_topics,
            kernel_initializer=tf.keras.initializers.Orthogonal(),
            name='latent_to_topics',
            activation='sigmoid',
            activity_regularizer=tf.keras.regularizers.L2(l2=self.latent_space_to_topic_proportions_coef),

        )
        
        self.z_log_sigma_layer = tf.keras.layers.Dense(self.n_topics,name='SigmaVector')
        
        self.dot_product_layer = tf.keras.layers.Dot(axes=1, name="tm_likelihood")

        self.decode_repeating_layer = tf.keras.layers.RepeatVector(self.max_len, name='RepeatLatentVector')
        self.decode_lstm_layer = tf.keras.layers.LSTM(self.latent_dim, name='DecoderLSTM1', return_sequences=True)

        self.decode_dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.alphabet_size, name='DecoderDense'))
        self.decode_softmax_layer = tf.keras.layers.Softmax(name="decoded_sequence") 

        self.kl_divergence_layer = tf.keras.layers.Layer(name="kl_divergence")
        self.correlation_reg_output_layer = tf.keras.layers.Layer(name="correlation_reg")
        self.label_prediction_layer = tf.keras.layers.Dense(1, name='label_likelihood', activation='sigmoid')
        
        # forward
        topic_proportions = self.sample_topic_proportions_layer(self.sample_input_layer)
        topic_proportions = tf.keras.activations.softmax(topic_proportions)
        topic_proportions = self.topic_proportions_reshape_layer(topic_proportions)

        encoded_seq_1 = self.encode_layer(self.seq_input_layer)
        encoded_seq_2 = self.encode_layer_2(self.seq_input_layer)
        encoded_seq = self.concat_layer([encoded_seq_1, encoded_seq_2])
        z_mean = self.pre_z_mean_layer_act(encoded_seq)
        z_mean = self.z_mean_layer(z_mean)
        z, z_log_sigma = self.sampling_layer(z_mean)

        seq_topic_probabilities = self.latent_space_to_topic_proportions_layer(z)

        self.seq_total_probability_output = self.dot_product_layer([topic_proportions, seq_topic_probabilities])
        self.label_output = self.label_prediction_layer(topic_proportions[:, :self.n_topics_signal]) # seq_total_probabilities

        kl_divergence = - 0.5 * tf.reduce_sum(1 + z_log_sigma - tf.math.square(z_mean) - tf.math.exp(z_log_sigma), axis=-1)
        self.kl_divergence_output = self.kl_divergence_layer(kl_divergence)
        
        decoded_sequence = self.decode_repeating_layer(z)
        decoded_sequence = self.decode_lstm_layer(decoded_sequence)
        decoded_sequence = self.decode_dense_layer(decoded_sequence)
        self.decoded_sequence_output = self.decode_softmax_layer(decoded_sequence)
        


        self.model = tf.keras.Model(
            inputs=[
                self.seq_input_layer,
                self.sample_input_layer
            ],
            outputs=[
                self.seq_total_probability_output,
                self.label_output,
                self.decoded_sequence_output,
                self.kl_divergence_output,
                # self.correlation_reg_output,
            ]
        )
        

    def compile_model(self, lr=1e-4):
        bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.model.compile(
            loss={
                "tm_likelihood": lambda x, y: bc_loss(x,y) * self.tm_likelihood_coef,
                "label_likelihood": lambda x, y: bc_loss(x,y) * self.label_likelihood_coef,
                "decoded_sequence": lambda x, y: AIRRTM.reconstruction_loss(x, y) * self.reconstruction_loss_coef,
                "kl_divergence": lambda x, y: (x+y) * self.kl_coef,
                # "correlation_reg": lambda x, y: y * self.decorrelation_regularizer_coef,
            },
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics={
                "tm_likelihood": ["acc", lambda x, y: (bc_loss(x,y) + np.log(0.5)) * 1e3],
                "label_likelihood": ["acc"],
                "decoded_sequence": ["acc"],#, lambda x, y: self.sample_topic_proportions_layer.losses[0]],
                "kl_divergence": [lambda x, y: y / self.latent_dim],
                # "correlation_reg": lambda x, y: y * self.decorrelation_regularizer_coef * 1000,
            }
        )

    def fit_model(self, batch_size, epochs, input_data, callbacks=[]):
        dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_labels, sample_sizes = input_data
        self.model.fit(
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
            
        )
    
    def get_topic_proportion_matrix(self):
        return self.sample_topic_proportions_layer.get_weights()[0]

    def from_tf_model(self, tf_model):
        self.model = tf_model
        self.sample_topic_proportions_layer = [l for l in tf_model.layers if l.name == 'topic_proportions'][0]
        self.latent_space_to_topic_proportions_layer = [l for l in tf_model.layers if l.name == 'latent_to_topics'][0]
        self.encode_layer = [l for l in tf_model.layers if isinstance(l, tf.keras.models.Sequential)][0]
        self.encode_layer_2 = [l for l in tf_model.layers if l.name == 'encoderLSTM'][0]
        self.z_mean_layer = [l for l in tf_model.layers if l.name == 'MeanVector'][0]
        self.pre_z_mean_layer_act = [l for l in tf_model.layers if isinstance(l, tf.keras.layers.PReLU)][0] 
        self.decode_repeating_layer = [l for l in tf_model.layers if l.name == 'RepeatLatentVector'][0]
        self.decode_lstm_layer = [l for l in tf_model.layers if l.name == 'DecoderLSTM1'][0]
        self.decode_dense_layer = [
            l for l in tf_model.layers if isinstance(l, tf.keras.layers.TimeDistributed)
        ][0]

    def encode_sequences(self, seqs):
        encoded_seq_1 = self.encode_layer(seqs)
        encoded_seq_2 = self.encode_layer_2(seqs)
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
        sample_topic_p_df = pd.DataFrame(sample_topic_p)
        topic_diffs = sample_topic_p_df.iloc[self.n_samples // 2:].mean(axis=0) - sample_topic_p_df.iloc[:self.n_samples // 2].mean(axis=0)
        best_topic_id = topic_diffs.argmax()
        topic_diffs_weights = topic_diffs.abs() / topic_diffs.abs().sum() * topic_diffs / topic_diffs.abs()
        predicted_topics = self.predict_topic_probs(seqs)
        signal_intensity = (predicted_topics  * topic_diffs_weights.to_numpy()).sum(axis=1)
        return signal_intensity

def seq_tensor_to_seq(seq_tensor):
    alphabet_encoder = LabelEncoder()
    alphabet_encoder.fit(alphabet)
    m = np.argmax(seq_tensor.numpy(), axis=-1)
    sequences = [''.join(alphabet_encoder.inverse_transform(s)) for s in m]
    return sequences


def get_default_model(max_len, n_samples):
    tm_coef = 0.995
    vae_coef = 1 - tm_coef

    tm_likelihood_coef=0.99
    label_likelihood_coef=(1.0 - tm_likelihood_coef)

    reconstruction_loss_coef=0.95
    kl_coef=(1.0 - reconstruction_loss_coef)

    decorrelation_regularizer_coef = 0.01

    n_topics_signal = 4
    n_topics_nonsignal = 4
    n_topics = n_topics_signal + n_topics_nonsignal
    latent_dim = 70
    airrtm_model = AIRRTM(
        max_len=max_len,
        n_samples=n_samples,
        n_topics_signal=n_topics_signal,
        n_topics_nonsignal=n_topics_nonsignal,
        latent_dim=latent_dim,
        decorrelation_regularizer_coef=decorrelation_regularizer_coef,
        entropy_regularizer_coef=0.0,
        topic_proportions_l2_coef=1e-4,
        tm_likelihood_coef=tm_coef * tm_likelihood_coef,
        label_likelihood_coef=tm_coef * label_likelihood_coef,
        reconstruction_loss_coef=vae_coef * reconstruction_loss_coef,
        kl_coef=vae_coef * kl_coef / latent_dim,
        latent_space_to_topic_proportions_coef=1e-3,
    )
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=50000,
        decay_rate=0.5
    )
    airrtm_model.compile_model(lr=lr_schedule)
    return airrtm_model


def load_model(model_file, max_len, n_samples):
    airrtm_model = get_default_model(max_len, n_samples)
    custom_objects = {
        "SamplingLayer": AIRRTM.SamplingLayer,
        "<lambda>": lambda M: AIRRTM.topic_proportions_reg(
            M,
            airrtm_model.entropy_regularizer_coef,
            airrtm_model.decorrelation_regularizer_coef,
            airrtm_model.topic_proportions_l2_coef
        ),
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        tf_model = tf.keras.models.load_model(model_file)
    airrtm_model.from_tf_model(tf_model)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=50000,
        decay_rate=0.5
    )
    airrtm_model.compile_model(lr=lr_schedule)
    return airrtm_model
