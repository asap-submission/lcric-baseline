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


## Some TODOs

- [x] Release the TQN baseline for LCric Dataset.
- [ ] Release the MemVit baseline for LCric Dataset based on its open source implementation.


## Citation

If you use this code etc., please cite the following paper:

```
@article{agarwal2023building,
  title={Building Scalable Video Understanding Benchmarks through Sports},
  author={Agarwal, Aniket and Zhang, Alex and Narasimhan, Karthik and Gilitschenski, Igor and Murahari, Vishvak and Kant, Yash},
  journal={arXiv preprint arXiv:2301.06866},
  year={2023}
}
```

TQN paper bibtex:

```
@inproceedings{zhangtqn,
  title={Temporal Query Networks for Fine-grained Video Understanding},
  author={Chuhan Zhang and Ankush Gputa and Andrew Zisserman},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```


If you have any question, please contact aagarwal@ma.iitr.ac.in .