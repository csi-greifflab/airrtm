import tensorflow as tf
import numpy as np

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
    for i in tqdm(range(n_samples // 2)):
        records = SeqIO.parse(f"{input_data_dir}/samples/{witness_rate}/{i}_neg.fasta", format="fasta")
        records = [str(s.seq) for s in records]
        if n_seq is not None:
            assert len(records) >= n_seq
            records = records[:n_seq]
        repertoires.append(records)
    for i in tqdm(range(n_samples // 2)):
        records = SeqIO.parse(f"{input_data_dir}/samples/{witness_rate}/{i}_pos.fasta", format="fasta")
        records = [str(s.seq) for s in records]
        if n_seq is not None:
            assert len(records) >= n_seq
            records = records[:n_seq]
        repertoires.append(records)
        continue
    if translate:
        repertoires_aa = [[str(Seq.Seq(s).translate()) for s in r] for r in tqdm(repertoires)]
    else:
        repertoires_aa = [[s for s in r] for r in repertoires]
    samples = [preprocess_seq_list(r, alphabet=alphabet, max_len=max_len) for r in tqdm(repertoires_aa)]
    return samples, repertoires_aa


def create_input_tensors(samples):
    n_samples = len(samples)
    n_seq = len(samples[0])
    dataset_repertoire_id = np.repeat(np.arange(n_samples), n_seq)
    dataset_tm_target = np.ones(n_samples * n_seq)
    dataset_kl_target = np.zeros(n_samples * n_seq)
    dataset_repertoire_label = np.concatenate([np.zeros(n_samples // 2 * n_seq), np.ones(n_samples // 2 * n_seq)])

    # adding negative examples
    dataset_seq = np.concatenate(samples * 2)
    dataset_repertoire_id = np.concatenate([dataset_repertoire_id, np.flip(dataset_repertoire_id)])
    dataset_tm_target = np.concatenate([np.ones(n_samples * n_seq), np.zeros(n_samples * n_seq)])
    dataset_kl_target = np.concatenate([np.zeros(n_samples * n_seq), np.zeros(n_samples * n_seq)])
    dataset_repertoire_label = np.concatenate([dataset_repertoire_label, np.flip(dataset_repertoire_label)])
    print(dataset_seq.shape, dataset_repertoire_id.shape, dataset_tm_target.shape, dataset_kl_target.shape, dataset_repertoire_label.shape)
    return dataset_seq, dataset_repertoire_id, dataset_tm_target, dataset_kl_target, dataset_repertoire_label
