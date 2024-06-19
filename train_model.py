
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-w', '--witness_rate', type=float, required=True)
parser.add_argument('-e', '--epoch', type=int, required=False)
parser.add_argument('-l', '--max_len', type=int, default=20)
parser.add_argument('-t', '--threads', type=int, default=8)
parser.add_argument('--checkpoint_dir', required=True)
parser.add_argument('--input_data_dir', required=True)
parser.add_argument('--use_vj', action='store_true')
parser.add_argument('--encoder', choices=['convolutional', 'lstm', 'transformer'], default='convolutional')

args = parser.parse_args()
witness_rate = args.witness_rate
warm_start_epoch = args.epoch

max_len = args.max_len
checkpoint_dir = args.checkpoint_dir
input_data_dir = args.input_data_dir
n_threads = args.threads
use_vj = args.use_vj

import os
os.environ['OMP_NUM_THREADS'] = f'{n_threads}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{n_threads}'
os.environ['MKL_NUM_THREADS'] = f'{n_threads}'
os.environ['VECLIB_MAXIMUM_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(
    n_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    n_threads
)


import gc
import importlib, sys
import json
import pickle
from pathlib import Path

from model import get_default_model, load_model
from utils import generate_random_sample_ids


witness_rate = args.witness_rate
warm_start_epoch = args.epoch
max_len = args.max_len
checkpoint_dir = args.checkpoint_dir
input_data_dir = args.input_data_dir
encoder = args.encoder

def main(
    witness_rate,
    warm_start_epoch,
    max_len,
    checkpoint_dir,
    input_data_dir,
    use_vj,
):
    with open(Path(input_data_dir) / 'input_tensors' / f'{witness_rate}.pickle', 'rb') as inp:
        input_data = pickle.load(inp)
    v_size = 0
    j_size = 0
    if use_vj:
        dataset_seq, vj_info, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes = input_data
        _, v_size, j_size = vj_info
    else:
        dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes = input_data

    if warm_start_epoch is None:
        airrtm_model = get_default_model(max_len, sample_labels, v_size, j_size, additional_encoder=encoder)
    else:
        airrtm_model = load_model(
            Path(checkpoint_dir) / f'model_{witness_rate}_epoch_{warm_start_epoch}.keras',
        )

    n_epochs = 30
    if warm_start_epoch is None:
        start_epoch = -1
    else:
        start_epoch = warm_start_epoch
    for epoch in range(start_epoch+1, start_epoch+1+n_epochs):
        print(f'===== epoch {epoch+1}/{n_epochs} =====')
        if use_vj:
            dataset_seq, vj_info, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes = input_data
        else:
            dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes = input_data
        dataset_repertoire_id = np.concatenate([
            dataset_repertoire_id[:dataset_repertoire_id.shape[0] // 2],
            generate_random_sample_ids(sample_sizes, sample_labels)
        ])
        if use_vj:
            input_data = dataset_seq, vj_info, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes
        else:
            input_data = dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label, sample_weights, sample_labels, sample_sizes
        gc.collect()
        airrtm_model.fit_model(batch_size=2048, epochs=1, input_data=input_data, callbacks=[])
        airrtm_model.save(
            Path(checkpoint_dir) / f'model_{witness_rate}_epoch_{epoch}.keras',
            save_format='keras',
        )

if __name__ == '__main__':
    main(
        witness_rate,
        warm_start_epoch,
        max_len,
        checkpoint_dir,
        input_data_dir,
        use_vj,
    )

