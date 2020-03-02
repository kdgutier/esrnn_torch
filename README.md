# TIME SERIES TRANSFER LEARNING

Repository intended to explore extensions to exponential smoothing RNN, and transferability of learned features.
Data obtained from M4 and M3 competitions.

<p float="center">
  <img src="results/m4_results.png" width="800" />
</p>

# Install

```python
pip install git+https://github.com/kdgutier/esrnn_torch@dev
```

# Conda environment
```console
local_user@local_host$ bash setup.sh
```
# Experiment Config File
Experiment Config File at configs/config_m4.yaml

# Run original ESRNN
```console
local_user@local_host$ conda activate esrnn_torch
local_user@local_host$ jupyter notebook m4_test.ipynb
local_user@local_host$ PYTHONPATH=. python src/M4_experiment.py --model_id 0
local_user@local_host$ PYTHONPATH=. python src/hyperpar_tunning_m4.py --id_min 0 --id_max 1 --dataset 'Quarterly' --gpu_id (optional)
```

# REFERENCES
## GENERAL
1. [History](https://robjhyndman.com/hyndsight/forecasting-competitions/)
2. [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
3. [Dilated RNN](https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf)
4. [Residual LSTM](https://arxiv.org/abs/1701.03360)
5. [Attention RNN](https://arxiv.org/abs/1704.02971)

## M4
1. [M4 Methods](https://github.com/M4Competition/M4-methods)
2. [M4 Hyndman](https://github.com/M4Competition/M4-methods/tree/master/245%20-%20pmontman)
3. [M4 Smyl](https://github.com/M4Competition/M4-methods/tree/master/118%20-%20slaweks17)
4. [M4 Competition Conclusions](https://rpubs.com/fotpetr/m4competition)
5. [M4 Data](https://github.com/M4Competition/M4-methods/tree/master/Dataset)

## M3
1. [M3 Competition Conclusions](https://www.sciencedirect.com/science/article/pii/S0169207011000616?via%3Dihub)
2. [M3 Data](http://www.neural-forecasting-competition.com/NN3/datasets.htm)
