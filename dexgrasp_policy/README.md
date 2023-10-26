# Dexterous Grasping Policy Learning Code for [CVPR2023]UniDexGrasp


This folder contains the dexterous grasping policy learning code for UniDexGrasp. For more information about this work, please check [our paper.](https://arxiv.org/abs/2303.00938)

## Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). We use the `Preview Release 3` version of IsaacGym in our experiment.

Please follow the steps below to perform the installation：


### 1. Create virtual environment
```bash
conda create -n dexgrasp python==3.8
conda activate dexgrasp
```

### 2. Install isaacgym
Once you have downloaded IsaacGym:
```bash
cd <PATH_TO_ISAACGYM_INSTALL_DIR>/python
pip install -e .
```
Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Please follow troubleshooting steps described in the Isaac Gym Preview Release 3/4
install instructions if you have any trouble running the samples.

### 3. Install dexgrasp
Once Isaac Gym is installed and samples work within your current python environment, install this repo from source code:
```bash
cd <PATH_TO_DEXGRASP_POLICY_DIR>
pip install -e .
```
cd DexGrasp-test
pip install -e .

### 4. Install pointnet2_ops
```bash
git clone git@github.com:erikwijmans/Pointnet2_PyTorch.git && cd pointnet2_ops_lib/
python setup.py install
```
## Dataset
We provide one training object instance example in `assets` folder. For trainig/evaluation datasets detailed information in our paper, please refer to `dexgrasp/cfg/train_set.yaml` (3200 object instances), `dexgrasp/cfg/test_set_seen_cat.yaml` (141 object instances) and `dexgrasp/cfg/test_set_unseen_cat.yaml` (100 object instances). You can modify the `object_code_dict` in `dexgrasp/cfg/shadow_hand_grasp.yaml` and  `dexgrasp/cfg/shadow_hand_random_load_vision.yaml` to change the training/testing object instances using the above dataset informations.

## Training/Evaluation
We provide two tasks: for the state-based policy task, please see `dexgrasp/tasks/shadow_hand_grasp.py`; for the vision-based policy tasks, in order to train on more objects within a certain GPU memory limit, we randomly load objects from the dataset in the beginning of each episode during training. please see `dexgrasp/tasks/shadow_hand_random_load_vision.py`.

Run the following lines in `dexgrasp` folder.

training state-based policy training using ppo:
```bash
bash script/run_train_ppo_state.sh 
```

training vision-based policy training using ppo:
```bash
bash script/run_train_ppo_vision.sh
```

training state to vision policy distillation using DAgger:
```bash
bash script/run_train_dagger.sh
```

For more provided args (e.g., backbone type, test mode), please check these scripts and `utils/config.py`.

## Acknowledgement
The code base used in this project is sourced from these repository:

[NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)

[PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands)

