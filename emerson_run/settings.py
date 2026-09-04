import os
import pathlib as pl

from dotenv import load_dotenv


DOTENV_FILENAME = ".env"
load_dotenv(DOTENV_FILENAME)

PROJECT_DIR = pl.Path(os.getenv("PROJECT_DIR"))
RAW_DATA_DIR = pl.Path(os.getenv("RAW_DATA_DIR"))
PROCESSED_DATA_DIR = pl.Path(os.getenv("PROCESSED_DATA_DIR"))
MODEL_DIR = pl.Path(os.getenv("MODEL_DIR"))
MAX_SEQUENCE_LENGTH = pl.Path(os.getenv("MAX_SEQUENCE_LENGTH"))
MIN_SEQUENCE_LENGTH = pl.Path(os.getenv("MIN_SEQUENCE_LENGTH"))
MIN_N_SEQUENCES = pl.Path(os.getenv("MIN_N_SEQUENCES"))
CONFIG_PATH = pl.Path(os.getenv("CONFIG_PATH"))
