"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym",
    "torch==1.13",
    "torchvision==0.14",
    "torchaudio==0.13",
    "matplotlib==3.5.1",
    "numpy==1.23.5",
    "tb-nightly",
    "tqdm",
    "ipdb",
    "pytorch_lightning",
    "opencv-python",
    "transforms3d",
    "addict",
    "yapf",
    "h5py",
    "sorcery",
    "psutil",
    "pynvml",
]

# Installation operation
setup(
    name="dexgrasp",
    author="test",
    version="0.1",
    description="Benchmark environments for Dexterous Grasping in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.8"],
    zip_safe=False,
)

# EOF
