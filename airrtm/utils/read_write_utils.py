import gzip
import pathlib as pl


def open_fasta_or_gz(input_path: pl.Path):
    if input_path.name.endswith(".fasta.gz"):
        reader = gzip.open(input_path, "rb")
    elif input_path.name.endswith(".fasta"):
        reader = open(input_path, "r")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    return reader


def read_fasta(input_path: pl.Path, output_format: str = "list"):
    if output_format == "list":
        seqs = []
        current_seq_list = []
        with open_fasta_or_gz(input_path) as fasta:
            for line in fasta:
                if line[0] == ">":
                    if current_seq_list != "":
                        seqs.append("".join(current_seq_list))
                        current_seq_list = []
                else:
                    current_seq_list.append(line.strip("\n"))
        if current_seq_list != "":
            seqs.append("".join(current_seq_list))
        return seqs
    else:
        raise ValueError(f"Unsupported requested format: {output_format}")
