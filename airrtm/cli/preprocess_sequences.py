import pathlib as pl

from argparse import ArgumentParser

import pandas as pd

from tqdm import tqdm

import airrtm.utils as au


METADATA_FILENAME = "metadata.csv"


def preprocess_single_repertoire():
    parser = ArgumentParser()
    parser.add_argument("--input_fasta", required=True, type=pl.Path)
    parser.add_argument("--output_path", required=True, type=pl.Path)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument(
        "--min_len",
        type=int,
        default=0,
        help="Filter out sequences with length less than minimum",
    )
    # parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()

    sequences = au.read_fasta(args.input_fasta)
    sequences = [s for s in sequences if s and len(s) >= args.min_len]
    sequences_dataset = au.SequenceDataset(
        sequence_data=sequences,
        alphabet_name="aa",
        max_length=args.max_length,
        device="cpu",
    )
    sequences_dataset.save(args.output_path)


def preprocess_from_csv():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--output_dir", required=True, type=pl.Path)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument(
        "--min_len",
        type=int,
        default=0,
        help="Filter out sequences with length less than minimum",
    )
    parser.add_argument("--min_n_sequences", type=int, default=0)
    parser.add_argument(
        "--sequence-colname", "--sequence_colname", type=str, default="cdr3_aa"
    )
    # parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()

    input_dir = pl.Path(args.input_dir)
    metadata_df = pd.read_csv(input_dir / METADATA_FILENAME)

    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    suffix = metadata_df["filename"].iloc[0].strip(".gz").split(".")[-1]
    if suffix == "fasta":
        separator = None
    elif suffix == "csv":
        separator = ","
    elif suffix == "tsv":
        separator = "\t"
    else:
        raise ValueError(f"Unsupported sequence file format {suffix}")
    for row_id in tqdm(range(metadata_df.shape[0])):
        input_filename = metadata_df.loc[row_id, "filename"]
        output_filename = _filename_to_pt(input_filename)
        sequences = (
            au.read_fasta(input_dir / input_filename)
            if separator is None
            else pd.read_csv(input_dir / input_filename, sep=separator)[
                args.sequence_colname
            ]
        )
        sequences = [s for s in sequences if s and len(s) >= args.min_len]
        if len(sequences) < args.min_n_sequences:
            print(
                f"Repertoire {input_filename} has {len(sequences)}<{args.min_n_sequences} sequences, skipping"
            )
            metadata_df.loc[row_id, "filename"] = None
            continue
        sequences_dataset = au.SequenceDataset(
            sequence_data=sequences,
            alphabet_name="aa",
            max_length=args.max_len,
            device="cpu",
        )
        sequences_dataset.save(output_dir / output_filename)

    metadata_df = metadata_df.loc[~metadata_df["filename"].isna()]
    metadata_df["filename"] = metadata_df["filename"].map(_filename_to_pt)
    metadata_df.to_csv(output_dir / METADATA_FILENAME, index=False)


def _filename_to_pt(filename: str) -> str:
    return ".".join(filename.split(".")[:-1] + ["pt"])
