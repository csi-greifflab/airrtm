# AIRRTM


This repository contains the code of the AIRRTM model from the article [Weakly supervised identification and generation of adaptive immune receptor sequences associated with immune disease status](https://www.biorxiv.org/content/10.1101/2023.09.24.558823v1). 

## Installation

### Pipenv
A locked Pipenv environment is provided with the repository.
To install the package, you need to run 
```shell
pipenv shell
pipenv install
```
The only prerequisites are havinng Pipenv and python3.9 installed on your machine.

### requirements.txt
Alternatively, you can use the `requirements.txt` file to install the dependencies in a *clean python3.9 environemnt*:
```shell
python -m pip install -r requirements.txt
```


## Data
Two synthetic datasets used in the article can be found in the corresponding [repository](https://github.com/csi-greifflab/airrtm_data). Note that you would need to unzip the data file for given dataset and witness rate (for example, [dataset S1, wr=0.0001](https://github.com/csi-greifflab/airrtm_data/blob/main/S1/samples/0.0001/data.zip)) before running training/prediction on them.

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

Additionally, optional columns `v_gene`, `j_gene` (to use with the `--use_vj` option), and `weight` (is used automatically when present) may be provided. 
```
cdr3_aa,v_gene,j_gene,weight
CATSRDVNTGELFF,TCRBV15-01,TCRBJ02-02,1
CASSPPGANVLTF,TCRBV11-02,TCRBJ02-06,1
CASSEYEQYF,TCRBV06-01,TCRBJ02-07,1
CASSLHEQYF,TCRBV11-02,TCRBJ02-07,9
CASSAATGATEAFF,TCRBV05-04,TCRBJ01-01,2
CASSPTGGHTEAFF,TCRBV05-04,TCRBJ01-01,2
CASSPQGAYNEQFF,TCRBV05-04,TCRBJ02-01,2
CASWGVNRGDAGYTF,TCRBV25-01,TCRBJ01-02,5042
CASSAQQGYSGNTIYF,TCRBV28-01,TCRBJ01-03,2247
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
python preprocess_data.py --input_data_dir INPUT_DATA_DIR -w WITNESS_RATE [--use_vj]
```

Then you can train the model by running
```
python train_model.py --input_data_dir INPUT_DATA_DIR -w WITNESS_RATE [-t THREADS] [--use_vj] --checkpoint_dir CHECKPOINT_DIR
```

And, with a trained model, make signal intensity predictions on sequences from an unseen repertoire, for example:
```
predict.py -i INPUT_FILE -o OUTPUT_FILE -m CHECKPOINT_DIR/model_0.005_epoch_9.keras [--use_vj]
```
The input csv file must be in the same format as the training files (i.e., it must have a column `cdr3_aa` with amino acid sequences).
Note that the `--use_vj` option must be used consistently, i.e., either by all three commands (`preprocess_data`, `train_model`, `predict`), or by none of them.



