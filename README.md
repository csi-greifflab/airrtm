# AIRRTM


This repository contains the code of the AIRRTM model from the article [Weakly supervised identification and generation of adaptive immune receptor sequences associated with immune disease status](https://www.biorxiv.org/content/10.1101/2023.09.24.558823v1). 

## Data
Two synthetic datasets used in the article can be found in the corresponding [repository](https://github.com/csi-greifflab/airrtm_data). 

The model can be trained on a dataset with the following structure:
```
INPUT_DATA_DIR
│
└───samples
    │
    └───WITNESS_RATE
    │   │   1.csv
    │   │   2.csv
    │   │   ...
    │   │   99.csv
    │   │   99.csv
    │   │   metadata.csv
```

Each of the data csv files must contain a column `cdr3_aa` with amino acid sequences.
```
$ head S1/samples/0.005/1.csv
cdr3_aa
CARDGRNTGIVGALTDPGMLLIS
CARGFGQPSSSW*SGWFDPW
CARDSSSWTT
CARDLRKGDYYDSSGYYYAFMMLLIS
CARERGRTVTVDYW
CARGCFFSMVRGVIITFRMLLIS
CARKFRWGRTGSAT
CARVVLLWFGELFDYGMDVW
CARDLIRGTLL*LL
```

The metadata file must contain columns `label` (i.e., repertoire label), `filename` and `split` (train/test). 
```
$ head S1/samples/0.005/metadata.csv
label,filename,split
1,0.csv,train
0,1.csv,train
1,2.csv,train
1,3.csv,train
```

## Usage

To train the model, one must first run `preprocess_data.py` on your dataset folder (structured as described above)
```
python preprocess_data.py --input_data_dir INPUT_DATA_DIR -w WITNESS_RATE
```

Then you can train the model by running
```
python train_model.py --input_data_dir INPUT_DATA_DIR -w WITNESS_RATE [-t THREADS] --checkpoint_dir CHECKPOINT_DIR
```

And, with a trained model, make signal intensity predictions on sequences from an unseen repertoire, for example:
```
predict.py -i INPUT_FILE -o OUTPUT_FILE -m CHECKPOINT_DIR/model_0.005_epoch_9.keras
```
The input csv file must be in the same format as the training files (i.e., it must have a column `cdr3_aa` with amino acid sequences).




