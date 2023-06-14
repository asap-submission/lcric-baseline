# Baseline for LCric Dataset

Implementation of Baseline for LCric dataset. The repository is heavily derived from the open source implementation of [Temporal Query Networks](https://github.com/Chuhanxx/Temporal_Query_Networks).

## Usage

The experiments are done on Python 3.9 version. Create an environment from the yml file

```
conda env create -f environment.yml
```

The directory structure for the repo should look as follows:

    .
    ├── configs                  
    │   ├── ...
    ├── data                        
    │   ├── ...               
    ├── engine                   
    │   ├── ...
    ├── models                   
    │   ├── ... 
    ├── scripts                 
    │   ├── ...
    ├── utils                   
    │   ├── ...   
    ├── dataset                     # place the videos and annotations downloaded from the lcric_downloader here
    │   ├── video
    │   ├── annotations                 
    ├── requirements.txt
    ├── README.md
    .

To start training the baseline model on 2 gpus, run the command from the main directory:

```
cd scripts

python train_1st_stage.py --dataset_config ../configs/cricket_first_stage.yaml --gpus 0,1
```
