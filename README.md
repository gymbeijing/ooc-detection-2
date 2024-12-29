# Contrastive Domain Adaptation with Test-time Training for Out-of-Context Detection
Yimeng Gu, Mengqi Zhang, Ignacio Castro, Shu Wu, Gareth Tyson

## Introduction
This repository contains code of our paper "Contrastive Domain Adaptation with Test-time Training for Out-of-Context Detection".

## Downloading the dataset
You can find the TwitterCOMMs dataset and the NewsCLIPpings dataset following the instructions in their repository.

## Set up the Environment
```
conda create -n venv_py38 python=3.8 -y
conda activate venv_py38
pip install -r requirements_py38.txt
```

## Running ConDA-TTT
For Twitter-COMMs:
```
(venv_py38) python -m trainers.train_conDATriplet --batch_size 256 --max_epochs 10 --tgt_topic climate --base_model blip-2 --loss_type simclr
```
For NewsCLIPpings:
```
(venv_py38) python -m trainers.train_conDATripletNews --batch_size 256 --max_epochs 10 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
```

If you find this repository or the paper useful, please cite it with:

```
@article{gu2024learning,
  title={Learning Domain-Invariant Features for Out-of-Context News Detection},
  author={Gu, Yimeng and Zhang, Mengqi and Castro, Ignacio and Wu, Shu and Tyson, Gareth},
  journal={arXiv preprint arXiv:2406.07430},
  year={2024}
}
```

For any inquiries about this repository or the paper, please contact Yimeng Gu (yimeng.gu@qmul.ac.uk).