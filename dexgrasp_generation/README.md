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

* Install the dependencies:
```commandline
conda install -y pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.6.2-py38_cu113_pyt1100.tar.bz2
pip install -r requirements.txt
cd thirdparty/pytorch_kinematics
pip install -e .
cd ../nflows
pip install -e .
cd ../
git clone https://github.com/wrc042/CSDF.git
cd CSDF
pip install -e .
cd ../../
```


## Data

1. Create a `data` folder under `dexgrasp_generation`:

```commandline
mkdir data
```
2. Download the data from [here](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/), and put them under `data`. Specifically, you need `mjcf` to build the ShadowHand, and `DFCData` contains the grasp labels.

<details>
  <summary> Click to see the file structure </summary>
  
  ```commandline
  UniDexGrasp
  ├── dexgrasp_generation
  │   ├── data
  │   │   ├── DFCdata
  │   │   └── mjcf
  │   └── ...
  ├── dexgrasp_policy
  └── ...
  ```
</details>

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
python ./network/train.py --config-name glow_joint_config \
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
* [CSDF](https://github.com/wrc042/CSDF)
* [nkolot's implementation of nflows](https://github.com/nkolot/nflows)
