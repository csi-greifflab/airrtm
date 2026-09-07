__all__ = [
    "METADATA_FILENAME",
    "GENE_VOCABULARY_FILENAME",
    "NT_ALPHABET",
    "AA_ALPHABET",
    "AA_NOSTOP_ALPHABET",
    "ALPHABETS",
]

NT_ALPHABET = tuple("ACGT")
AA_ALPHABET = tuple("RHKDESTNQCGPAVILMFYW*")
AA_NOSTOP_ALPHABET = AA_ALPHABET[:-1]
ALPHABETS = {
    "nt": NT_ALPHABET,
    "aa": AA_ALPHABET,
}

#: Name of the per-dataset metadata table (columns: label, filename, split, ...).
METADATA_FILENAME = "metadata.csv"

#: Written next to the datasets so training can rebuild the V/J vocabularies.
GENE_VOCABULARY_FILENAME = "gene_vocabulary.pt"
