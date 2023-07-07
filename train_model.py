import os
n_threads = 8
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"


import gc
import importlib, sys
import json
import pickle 

from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument('-w', '--witness_rate', type=float, required=True)
parser.add_argument('-e', '--epoch', type=int, required=False)
parser.add_argument('-l', '--max_len', type=int, default=20)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_seqs', type=int, default=80 * 1000)
parser.add_argument('--kmers', required=True)
parser.add_argument('--checkpoint_dir', required=True)
parser.add_argument('--input_data_dir', required=True)

args = parser.parse_args()
witness_rate = args.witness_rate
warm_start_epoch = args.epoch

max_len = args.max_len
n_samples = args.n_samples
n_seq = args.n_seqs
kmers = args.kmers.split(',') #["QGD", "IKL", "ENQ", "SPF"] or ["DPM"]
checkpoint_dir = args.checkpoint_dir
input_data_dir = args.input_data_dir


def main():
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

    if warm_start_epoch is not None:
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
            tf_model = tf.keras.models.load_model(Path(checkpoint_dir) / f'model_{witness_rate}_epoch_{warm_start_epoch}.hdf5')
        airrtm_model.load_model(tf_model)
        airrtm_model.compile_model(lr=lr_schedule)
        
    with open(Path(input_data_dir) / f'{witness_rate}.pickle', 'rb') as inp:
        input_data = pickle.load(inp)

    n_epochs = 20
    if warm_start_epoch is None:
        start_epoch = -1
    else:
        start_epoch = warm_start_epoch
    for epoch in range(start_epoch+1, start_epoch+1+n_epochs):
        print(f"===== epoch {epoch+1}/{n_epochs} =====")
        dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label = input_data
        dataset_repertoire_id = np.repeat(np.arange(n_samples), n_seq)
        reverse_base = 100 - ((dataset_repertoire_id // 50) + 1) * 50
        random_repertoire_id = np.random.randint(0, 50, size=dataset_repertoire_id.shape)
        random_repertoire_id = random_repertoire_id + reverse_base
        dataset_repertoire_id = np.concatenate([dataset_repertoire_id, np.flip(dataset_repertoire_id)])
        input_data = dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label
        gc.collect()
        airrtm_model.fit_model(batch_size=2048, epochs=1, input_data=input_data, callbacks=[])
        airrtm_model.model.save(Path(checkpoint_dir) / f'model_{witness_rate}_epoch_{epoch}.hdf5')

if __name__ == '__main__':
    main()

