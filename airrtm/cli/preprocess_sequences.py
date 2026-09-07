import pathlib as pl

from argparse import ArgumentParser

import pandas as pd
import torch

from tqdm import tqdm

import airrtm.utils as au


# Re-exported for backwards compatibility; the definitions live in utils.constants.
METADATA_FILENAME = au.METADATA_FILENAME
GENE_VOCABULARY_FILENAME = au.GENE_VOCABULARY_FILENAME

SEPARATOR_BY_SUFFIX = {"fasta": None, "csv": ",", "tsv": "\t"}


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
    args = parser.parse_args()

    sequences = au.read_fasta(args.input_fasta)
    sequences = [s for s in sequences if s and len(s) >= args.min_len]
    sequences_dataset = au.SequenceDataset(
        sequence_data=sequences,
        alphabet_name="aa",
        max_length=args.max_len,
        device="cpu",
    )
    sequences_dataset.truncate(args.max_len)
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
    parser.add_argument(
        "--count-colname",
        "--count_colname",
        type=str,
        default="duplicate_count",
        help="Clonal abundance column. Ignored when absent from the input files.",
    )
    parser.add_argument(
        "--v-colname", "--v_colname", type=str, default="v_call",
    )
    parser.add_argument(
        "--j-colname", "--j_colname", type=str, default="j_call",
    )
    parser.add_argument(
        "--no-counts",
        action="store_true",
        help="Discard clonal abundances even when the column is present",
    )
    parser.add_argument(
        "--no-vj",
        action="store_true",
        help="Discard V/J gene assignments even when the columns are present",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep duplicate rows instead of collapsing them into weighted ones",
    )
    args = parser.parse_args()

    input_dir = pl.Path(args.input_dir)
    metadata_df = pd.read_csv(input_dir / METADATA_FILENAME)

    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    suffix = metadata_df["filename"].iloc[0].strip(".gz").split(".")[-1]
    if suffix not in SEPARATOR_BY_SUFFIX:
        raise ValueError(f"Unsupported sequence file format {suffix}")
    separator = SEPARATOR_BY_SUFFIX[suffix]

    filenames = list(metadata_df["filename"])
    v_genes, j_genes = (None, None)
    if separator is not None and not args.no_vj:
        v_genes, j_genes = _collect_gene_vocabularies(
            input_dir, filenames, separator, args.v_colname, args.j_colname
        )
        if v_genes is not None:
            torch.save(
                {"v_genes": v_genes, "j_genes": j_genes},
                output_dir / GENE_VOCABULARY_FILENAME,
            )

    for row_id in tqdm(range(metadata_df.shape[0])):
        input_filename = metadata_df.loc[row_id, "filename"]
        output_filename = _filename_to_pt(input_filename)

        if separator is None:
            sequences = au.read_fasta(input_dir / input_filename)
            sequences = [s for s in sequences if s and len(s) >= args.min_len]
            weights, v_ids, j_ids = None, None, None
        else:
            sequences, weights, v_ids, j_ids = _read_table(
                path=input_dir / input_filename,
                separator=separator,
                sequence_colname=args.sequence_colname,
                count_colname=None if args.no_counts else args.count_colname,
                v_colname=None if v_genes is None else args.v_colname,
                j_colname=None if v_genes is None else args.j_colname,
                v_genes=v_genes,
                j_genes=j_genes,
                min_len=args.min_len,
                dedup=not args.no_dedup,
            )

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
            weights=weights,
            v_ids=v_ids,
            j_ids=j_ids,
            v_genes=v_genes,
            j_genes=j_genes,
        )
        # `pad` is a no-op when the data is already longer than `max_len`, so truncate
        # explicitly: every repertoire must end up with exactly the same `max_length`,
        # otherwise the training loop cannot concatenate them into one batch.
        sequences_dataset.truncate(args.max_len)
        sequences_dataset.save(output_dir / output_filename)

    metadata_df = metadata_df.loc[~metadata_df["filename"].isna()]
    metadata_df["filename"] = metadata_df["filename"].map(_filename_to_pt)
    metadata_df.to_csv(output_dir / METADATA_FILENAME, index=False)


def _read_table(
    *,
    path: pl.Path,
    separator: str,
    sequence_colname: str,
    count_colname: str | None,
    v_colname: str | None,
    j_colname: str | None,
    v_genes: tuple[str, ...] | None,
    j_genes: tuple[str, ...] | None,
    min_len: int,
    dedup: bool,
) -> tuple[list[str], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Read one repertoire, optionally collapsing duplicate clonotypes into weights."""
    header = pd.read_csv(path, sep=separator, nrows=0).columns
    usecols = [sequence_colname]
    use_counts = count_colname is not None and count_colname in header
    use_vj = (
        v_colname is not None
        and j_colname is not None
        and v_colname in header
        and j_colname in header
    )
    if use_counts:
        usecols.append(count_colname)
    if use_vj:
        usecols.extend([v_colname, j_colname])

    dtypes = {c: str for c in usecols if c in (v_colname, j_colname)}
    df = pd.read_csv(path, sep=separator, usecols=usecols, dtype=dtypes)
    df = df.loc[df[sequence_colname].notna()]
    df = df.loc[df[sequence_colname].str.len() >= min_len]
    df["_count"] = (
        df[count_colname].fillna(1).astype("float32") if use_counts else 1.0
    )
    if use_vj:
        df["_v"] = df[v_colname].fillna("").astype(str)
        df["_j"] = df[j_colname].fillna("").astype(str)

    if dedup:
        group_keys = [sequence_colname] + (["_v", "_j"] if use_vj else [])
        df = df.groupby(group_keys, as_index=False, sort=False)["_count"].sum()

    sequences = df[sequence_colname].tolist()
    weights = torch.tensor(df["_count"].to_numpy(), dtype=torch.float32)
    v_ids, j_ids = None, None
    if use_vj:
        v_index = {gene: i for i, gene in enumerate(v_genes)}
        j_index = {gene: i for i, gene in enumerate(j_genes)}
        v_ids = torch.tensor(
            [v_index.get(g, len(v_genes) - 1) for g in df["_v"]], dtype=torch.int64
        )
        j_ids = torch.tensor(
            [j_index.get(g, len(j_genes) - 1) for g in df["_j"]], dtype=torch.int64
        )
    return sequences, weights, v_ids, j_ids


def _collect_gene_vocabularies(
    input_dir: pl.Path,
    filenames: list[str],
    separator: str,
    v_colname: str,
    j_colname: str,
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """One cheap pass over the V/J columns to build a vocabulary shared by all repertoires.

    Gene ids must mean the same thing in every repertoire, so they cannot be assigned
    per file. The last entry of each vocabulary is a catch-all for unseen genes.
    """
    header = pd.read_csv(input_dir / filenames[0], sep=separator, nrows=0).columns
    if v_colname not in header or j_colname not in header:
        return None, None

    v_seen, j_seen = set(), set()
    for filename in tqdm(filenames, desc="collecting V/J vocabulary"):
        df = pd.read_csv(
            input_dir / filename,
            sep=separator,
            usecols=[v_colname, j_colname],
            dtype={v_colname: str, j_colname: str},
        )
        v_seen.update(df[v_colname].fillna("").astype(str).unique().tolist())
        j_seen.update(df[j_colname].fillna("").astype(str).unique().tolist())
    v_genes = tuple(sorted(v_seen)) + ("__unknown__",)
    j_genes = tuple(sorted(j_seen)) + ("__unknown__",)
    print(f"V gene vocabulary: {len(v_genes)}, J gene vocabulary: {len(j_genes)}")
    print(f"  V examples: {v_genes[:3]}\n  J examples: {j_genes[:3]}")
    # A real V/J column has tens of distinct genes. Some AIRR exports put bare allele
    # suffixes in v_call/j_call (Emerson's do -- the genes are in v_resolved/j_resolved),
    # and embedding that column would be silently meaningless.
    for name, genes in (("V", v_genes), ("J", j_genes)):
        if len(genes) < 6:
            raise ValueError(
                f"{name} gene column looks wrong: only {len(genes) - 1} distinct values "
                f"{genes[:5]}. Point --v-colname/--j-colname at the real gene column, "
                f"or pass --no-vj."
            )
    return v_genes, j_genes


def _filename_to_pt(filename: str) -> str:
    return ".".join(filename.split(".")[:-1] + ["pt"])
