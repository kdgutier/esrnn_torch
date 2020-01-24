# ts_transfer packages
conda create --name esrnn_torch python=3.6
#source ~/anaconda3/etc/profile.d/conda.sh
source ~/miniconda/etc/profile.d/conda.sh
conda activate esrnn_torch

# basic
conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2

# visualization
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0

# pytorch
#conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch
conda install pytorch=1.2.0 -c pytorch
#conda install -c anaconda tensorflow=1.13.1

conda install -c conda-forge jupyterlab
ipython kernel install --user --name=esrnn_torch
conda install -c anaconda pylint
conda install -c anaconda pyyaml
conda deactivate
