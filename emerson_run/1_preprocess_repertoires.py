#! /bin/env python

import os

from settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    MIN_N_SEQUENCES,
)

cmd = [
    "preprocess-from-csv",
    "--input_dir",
    RAW_DATA_DIR,
    "--output_dir",
    PROCESSED_DATA_DIR,
    "--max_len",
    MAX_SEQUENCE_LENGTH,
    "--min_len",
    MIN_SEQUENCE_LENGTH,
    "--min_n_sequences",
    MIN_N_SEQUENCES,
]
cmd_str = " ".join((str(s) for s in cmd))
print(cmd_str)
os.system(cmd_str)
