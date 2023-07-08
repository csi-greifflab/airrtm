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
    alphabet_size = len(alphabet)
    label_encoder = LabelEncoder()
    label_encoder.fit(alphabet)
    seqs = [seq[:max_len].upper() for seq in seqs]
    array = np.array([label_encoder.transform(list(seq) + [alphabet[0]] * (max_len - len(seq))) for seq in seqs])
    res = tf.one_hot(array, depth=alphabet_size).numpy()
    return res


def load_data(input_data_dir, witness_rate, max_len, alphabet, n_samples, n_seq=None, translate=True):
    repertoires = []
    signal_seqs_ids = []
    for i in tqdm(range(n_samples)):
        records = SeqIO.parse(f'{input_data_dir}/samples/{witness_rate}/{i}.fasta', format='fasta')
        records = [str(s.seq) for s in records]
        if n_seq is not None:
            assert len(records) >= n_seq
            records = records[:n_seq]
        repertoires.append(records)
    if translate:
        repertoires_aa = [[str(Seq.Seq(s).translate()) for s in r] for r in tqdm(repertoires)]
    else:
        repertoires_aa = [[s for s in r] for r in repertoires]
    samples = [preprocess_seq_list(r, alphabet=alphabet, max_len=max_len) for r in tqdm(repertoires_aa)]
    sample_labels_df = pd.read_csv(f'{input_data_dir}/samples/{witness_rate}/metadata.csv')
    sample_labels = sample_labels_df['label'].to_list()
    return samples, repertoires_aa, sample_labels


def create_input_tensors(samples, sample_labels):
    n_samples = len(samples)
    sample_sizes = [sample.shape[0] for sample in samples]
    total_size = sum(sample_sizes)

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

    # adding negative examples
    dataset_seq_2 = np.concatenate(samples * 2)
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
        1 - dataset_repertoire_label # repertoire_id comes from the opposite class
    ])
    print(dataset_seq_2.shape, dataset_repertoire_id_2.shape, dataset_tm_target_2.shape, dataset_kl_target_2.shape, dataset_repertoire_label_2.shape)
    return dataset_seq_2, dataset_repertoire_id_2, dataset_tm_target_2, dataset_kl_target_2, dataset_repertoire_label_2, sample_labels, sample_sizes


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

