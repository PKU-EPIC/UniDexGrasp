# Dexterous Grasping Proposal Generation Code for [CVPR2023]UniDexGrasp


This folder contains the dexterous grasping proposal generation code for UniDexGrasp. For more information about this work, please check [our paper.](https://arxiv.org/abs/2303.00938)


## Installation

Our code is tested on (Ubuntu 20.04).

* Clone this repository:
```commandline
git clone https://github.com/PKU-EPIC/UniDexGrasp.git
cd UniDexGrasp
```

* Create a [conda](https://www.anaconda.com/) environment and activate it:
```commandline
conda create -n unidexgrasp python=3.8
conda activate unidexgrasp
```

* Install the dependencies: (**TODO**)
```commandline
pip install -r requirements.txt
```

**TODO**: other manually-installed dependencies


## Data

**TODO**.


## Training

### GraspIPDF

```commandline
python ./network/train.py --config-name ipdf_config \
                          --exp-dir ./ipdf_train
```

### GraspGlow

```commandline
python ./network/train.py --config-name glow_config \
                          --exp-dir ./glow_train
```

### ContactNet

```commandline
python ./network/train.py --config-name cm_net_config \
                          --exp-dir ./cm_net_train
```

## Evaluation

```commandline
python ./network/eval.py  --config-name eval_config \
                          --exp-dir=./eval
```



## Acknowledgements

* [PointNet++](https://github.com/rusty1s/pytorch_geometric)
* [Implicit PDF](https://github.com/google-research/google-research/tree/master/implicit_pdf)
* [ProHMR](https://github.com/nkolot/nflows)