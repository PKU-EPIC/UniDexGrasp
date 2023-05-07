# UniDexGrasp
Official code for "**UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy**" *(CVPR 2023)*

[Project Page](https://pku-epic.github.io/UniDexGrasp/) | [Paper](https://arxiv.org/abs/2303.00938)

Coming soon.

![image](./images/teaser.png)


## Installation

Our code is tested on (Ubuntu xx.xx **TODO**).

* Clone this repository:
```commandline
git clone https://github.com/PKU-EPIC/UniDexGrasp.git
cd UniDexGrasp
```

* Create a [conda](https://www.anaconda.com/) environment and activate it:
```commandline
conda create -n unidexgrasp python=TODO
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
python ./network/train.py TODO
```

### Execution Policy Learning for Dexterous Grasping

Please see [README](https://github.com/PKU-EPIC/UniDexGrasp/blob/main/dexgrasp_policy/README.md) in `dexgrasp_policy` folder.

## Evaluation

**TODO**.


## Citation

```
@article{xu2023unidexgrasp,
  title={UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy},
  author={Xu, Yinzhen and Wan, Weikang and Zhang, Jialiang and Liu, Haoran and Shan, Zikang and Shen, Hao and Wang, Ruicheng and Geng, Haoran and Weng, Yijia and Chen, Jiayi and others},
  journal={arXiv preprint arXiv:2303.00938},
  year={2023}
}
```


## Acknowledgements

* [PointNet++](https://github.com/rusty1s/pytorch_geometric)
* [Implicit PDF](https://github.com/google-research/google-research/tree/master/implicit_pdf)
