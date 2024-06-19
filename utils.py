import json

import tensorflow as tf
import numpy as np
import pandas as pd

from Bio import SeqIO, Seq
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


alphabet_aa = [
    'O', '*',
    'I', 'V', 'D', 'W', 'Q', 'F', 'L', 'S', 'H', 'C', 'Y', 'A', 'K', 'G', 'N', 'T', 'R', 'E', 'P', 'M'
]
alphabet = alphabet_aa
alphabet_size = len(alphabet)


def check_kmers(s, kmers):
    for kmer in kmers:
        if kmer in s:
            return True
    return False


def check_kmers_2(s, kmers):
    for i, kmer in enumerate(kmers):
        if kmer in s:
            return i
    return -1


def preprocess_seq_list(seqs, max_len, alphabet):
    label_encoder = LabelEncoder()
    label_encoder.fit(alphabet)
    seqs = [seq[:max_len].upper() for seq in seqs]
    array = np.array([label_encoder.transform(list(seq) + [alphabet[0]] * (max_len - len(seq))) for seq in seqs])
    res = array.reshape(-1, max_len, 1)
    # res = tf.one_hot(array, depth=alphabet_size).numpy()
    return res


def preprocess_vg_gene_list(vj_lists, vj_dict):
    label_encoder = LabelEncoder()
    label_encoder.fit(vj_dict['v'])
    v_array = label_encoder.transform(vj_lists['v'])
    label_encoder.fit(vj_dict['j'])
    j_array = label_encoder.transform(vj_lists['j'])
    return np.array([v_array, j_array]).T


def load_data(
    input_data_dir, witness_rate, max_len, min_len, alphabet,
    # n_samples,
    n_seq=None,
    translate=True,
    use_vj=False,
):
    metadata_df = pd.read_csv(f'{input_data_dir}/samples/{witness_rate}/metadata.csv')
    metadata_df = metadata_df.loc[metadata_df['split'] == 'train'].reset_index(drop=True)
    n_samples = metadata_df.shape[0]
    print(f'n_samples={n_samples}')
    repertoires = []
    if use_vj:
        with open(f'{input_data_dir}/samples/{witness_rate}/metadata_vj_gene_list.json') as inp:
            vj_dict = json.load(inp)
        repertoire_vj_genes = []

    for row_id in tqdm(range(n_samples)):
        row = metadata_df.iloc[row_id]
        filename = row['filename']
        if filename.endswith('.csv'):
            sep = ','
        elif filename.endswith('.tsv'):
            sep = '\t'
        else:
            raise ValueError(f'Unknown format {filename}')
        samples_df = pd.read_csv(f'{input_data_dir}/samples/{witness_rate}/{filename}', sep=sep)
        if n_seq is not None:
            if samples_df.shape[0] < n_seq:
                print('found only {samples_df.shape[0]} sequences when specified n_seq={n_seq}')
            samples_df = samples_df.iloc[:n_seq]
        records = samples_df['cdr3_aa'].fillna('AAA').to_list()
        repertoires.append(records)
        if use_vj:
            repertoire_vj_genes.append({'v': samples_df['v_gene'], 'j': samples_df['j_gene']})

    if translate:
        repertoires_aa = [[str(Seq.Seq(s).translate()) for s in r] for r in tqdm(repertoires)]
    else:
        repertoires_aa = [[s for s in r] for r in repertoires]
    repertoires_aa = [[s for s in r if len(s) >= min_len] for r in repertoires_aa]
    samples = [preprocess_seq_list(r, alphabet=alphabet, max_len=max_len) for r in tqdm(repertoires_aa)]
    sample_labels = metadata_df['label'].to_list()
    if use_vj:
        sample_vj_genes = [preprocess_vg_gene_list(vj_lists, vj_dict=vj_dict) for vj_lists in tqdm(repertoire_vj_genes)]
        v_size = len(vj_dict['v'])
        j_size = len(vj_dict['j'])
        return samples, repertoires_aa, sample_labels, sample_vj_genes, v_size, j_size
    return samples, repertoires_aa, sample_labels


def create_input_tensors(
        samples: list[np.array],
        sample_labels: list[int],
        sample_vj_genes: list[np.array] = None,
        v_size: int = 0,
        j_size: int = 0
    ):
    use_vj = (sample_vj_genes is not None)
    n_samples = len(samples)
    sample_sizes = [sample.shape[0] for sample in samples]
    total_size = sum(sample_sizes)
    sample_labels = sample_labels

    dataset_repertoire_id = np.concatenate([
        np.full(
            fill_value=sample_id,
            shape=sample_size
        ) for sample_id, sample_size in enumerate(sample_sizes)
    ])
    dataset_tm_target = np.ones(total_size)
    dataset_kl_target = np.zeros(total_size)
    dataset_repertoire_label = np.concatenate([
        np.full(
            fill_value=sample_label,
            shape=sample_size
        ) for sample_size, sample_label in zip(sample_sizes, sample_labels)
    ])

    neg_class_weight = np.mean(sample_labels)
    pos_class_weight = 1 - neg_class_weight
    normalizer = 2 * neg_class_weight * pos_class_weight
    neg_class_weight /=  normalizer
    pos_class_weight /=  normalizer
    sample_weights = np.concatenate([
        np.full(
            fill_value=1 / sample_size * (pos_class_weight if sample_label else neg_class_weight),
            shape=sample_size
        ) for sample_size, sample_label in zip(sample_sizes, sample_labels)
    ])

    # adding negative examples
    dataset_seq_2 = np.concatenate(samples * 2)
    if use_vj:
        dataset_vj_2 = np.concatenate(sample_vj_genes * 2)
    dataset_repertoire_id_2 = np.concatenate([
        dataset_repertoire_id,
        generate_random_sample_ids(sample_sizes, sample_labels)
    ])
    dataset_tm_target_2 = np.concatenate([
        np.ones_like(dataset_tm_target), # come from the sample
        np.zeros_like(dataset_tm_target) # do not come from the sample
    ])
    dataset_kl_target_2 = np.zeros_like(dataset_tm_target_2)
    dataset_repertoire_label_2 = np.concatenate([
        dataset_repertoire_label, 
        # dataset_repertoire_label # repertoire_id comes from the opposite class
        1 - dataset_repertoire_label # repertoire_id comes from the opposite class
    ])
    sample_weights_2 = np.concatenate([
        sample_weights, 
        sample_weights,
    ]) *  sample_weights.shape[0] / n_samples
    if use_vj:
        print(dataset_seq_2.shape, dataset_vj_2.shape, dataset_repertoire_id_2.shape, dataset_tm_target_2.shape, dataset_kl_target_2.shape, dataset_repertoire_label_2.shape, sample_weights_2.shape, sample_weights_2.sum())
        return dataset_seq_2, (dataset_vj_2, v_size, j_size), dataset_repertoire_id_2, dataset_tm_target_2, dataset_kl_target_2, dataset_repertoire_label_2, sample_weights_2, sample_labels, sample_sizes    
    print(dataset_seq_2.shape, dataset_repertoire_id_2.shape, dataset_tm_target_2.shape, dataset_kl_target_2.shape, dataset_repertoire_label_2.shape, sample_weights_2.shape, sample_weights_2.sum())
    return dataset_seq_2, dataset_repertoire_id_2, dataset_tm_target_2, dataset_kl_target_2, dataset_repertoire_label_2, sample_weights_2, sample_labels, sample_sizes


def generate_random_sample_ids(sample_sizes, sample_labels):
    pos_labels = [i for i, label in enumerate(sample_labels) if label]
    neg_labels = [i for i, label in enumerate(sample_labels) if not label]
    dataset_repertoire_id = np.concatenate([
        np.random.choice(
            neg_labels if sample_label else pos_labels,
            size=sample_size
        ) for sample_size, sample_label in zip(sample_sizes, sample_labels)
    ])
    return dataset_repertoire_id

