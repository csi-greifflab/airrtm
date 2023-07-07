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

import pandas as pd

from Bio import SeqIO, Seq

from model import load_model
from utils import alphabet, preprocess_seq_list


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_fasta', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-m', '--model_file', required=True)
    parser.add_argument('-l', '--max_len', type=int, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--translate', action='store_true')

    args = parser.parse_args()
    input_fasta = args.input_fasta
    output_file = args.output_file
    model_file = args.model_file
    max_len = args.max_len
    n_samples = args.n_samples
    translate = args.translate

    airrtm_model = load_model(model_file, max_len, n_samples)
    records = SeqIO.parse(input_fasta, format="fasta")
    records = [str(s.seq) for s in records]
    if translate:
        records_aa = [str(Seq.Seq(s).translate()) for s in records]
    else:
        records_aa = records
    dataset_seq = preprocess_seq_list(records_aa, alphabet=alphabet, max_len=max_len)
    signal_intensity = airrtm_model.predict_signal_intensity(dataset_seq)
    signal_intensity_df = pd.DataFrame({
        'CDR3_aa': records_aa,
        'predicted_signal_intensity': signal_intensity,
    }).sort_values(by='predicted_signal_intensity', ascending=False)
    signal_intensity_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()

