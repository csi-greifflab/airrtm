import os
n_threads = 64
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(
    n_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    n_threads
)
tf.config.set_soft_device_placement(True)

import json
import logging

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from Bio import SeqIO, Seq
from tqdm import tqdm

from model import load_model
from utils import alphabet, preprocess_seq_list, preprocess_vg_gene_list


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-m', '--model_file', required=True)
    parser.add_argument('-l', '--max_len', type=int, required=True)
    parser.add_argument('--translate', action='store_true')
    parser.add_argument('--from_metadata', action='store_true')
    parser.add_argument('--use_vj', action='store_true')


    args = parser.parse_args()
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    model_file = args.model_file
    max_len = args.max_len
    translate = args.translate
    use_vj = args.use_vj

    output_dir.mkdir(parents=True, exist_ok=True)
    
    airrtm_model = load_model(model_file)
    if not args.from_metadata and input_file.name == 'metadata.csv':
        raise ValueError(f'--from_metadata flag should be passed if input_file is a metadata file')
    if args.from_metadata:
        metadata_path = input_file
        meta_df = pd.read_csv(metadata_path)
        filenames = meta_df['filename'].to_list()
        files = tqdm((metadata_path.parent / f for f in filenames), total=meta_df.shape[0])
        if use_vj:
            with open(metadata_path.parent / 'metadata_vj_gene_list.json') as inp:
                vj_dict = json.load(inp)
            
        
    else:
        files = [input_file]
    for f in files:
        if f.suffix == '.fasta':
            records = SeqIO.parse(f, format='fasta')
            records = [str(s.seq) for s in records]
        elif f.suffix == '.tsv':
            samples_df = pd.read_csv(f, sep='\t')
            if samples_df['cdr3_aa'].isna().any():
                logging.warning(f"Found {samples_df['cdr3_aa'].isna().sum()} NA rows, skipping")
                samples_df = samples_df.loc[~samples_df['cdr3_aa'].isna()].reset_index(drop=True)
            records = samples_df['cdr3_aa'].to_list()
        elif f.suffix == '.csv':
            samples_df = pd.read_csv(f, sep=',')
            if samples_df['cdr3_aa'].isna().any():
                logging.warning(f"Found {samples_df['cdr3_aa'].isna().sum()} NA rows, skipping")
                samples_df = samples_df.loc[~samples_df['cdr3_aa'].isna()].reset_index(drop=True)
            records = samples_df['cdr3_aa'].to_list()
        else:
            raise ValueError(f'Unknown input file format: {f.suffix}')
        if translate:
            records_aa = [str(Seq.Seq(s).translate()) for s in records]
        else:
            records_aa = records
        dataset_seq = preprocess_seq_list(records_aa, alphabet=alphabet, max_len=max_len)
        dataset_vj = None
        if use_vj:
            dataset_vj = preprocess_vg_gene_list(
                {'v': samples_df['v_gene'], 'j': samples_df['j_gene']},
                vj_dict=vj_dict
            )
        topic_probs = airrtm_model.predict_topic_probs(dataset_seq, dataset_vj)
        signal_intensity = airrtm_model.predict_signal_intensity(dataset_seq, dataset_vj)

        signal_intensity_df = pd.DataFrame({
            'cdr3_aa': records_aa,
            'signal_intensity': signal_intensity,
        })
        for topic_id in range(topic_probs.shape[1]):
            signal_intensity_df[f'topic_{topic_id}_prob'] = topic_probs[:, topic_id]
        signal_intensity_df = signal_intensity_df.sort_values(by='signal_intensity', ascending=False)
        output_file = output_dir / f.with_suffix('.csv').name
        signal_intensity_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()

