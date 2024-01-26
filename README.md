# Overlay Left Atrium Mantle-Free for Semi-Supervised Medical Image Segmentation
by Jiacheng Liu[0009−0000−3354−6139], Wenhua Qian, Jinde Cao and Peng Liu.
<img width="1483" alt="截屏2024-01-26 16 51 23" src="https://github.com/vigilliu/OMF/assets/129838909/9eacab39-29ae-43af-b0e9-c03b21de8c76">

## Introduction
Official code for "[Overlay Left Atrium Mantle-Free for Semi-Supervised Medical Image Segmentation](https://arxiv.org/)".

## Overview of our method
<img width="1017" alt="截屏2024-01-26 16 52 10" src="https://github.com/vigilliu/OMF/assets/129838909/9ddc8c07-cf38-4e4f-8c34-e01ca2fb1a3d">

## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.6.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

## Usage
Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data).

To train a model,
```
python ./code/LA_BCP_train.py --exp=OMF_lab8 --base_lr=0.1 --labelnum=8  #for LA training
python ./code/LA_BCP_train.py --exp=OMF_lab16 --base_lr=0.1 --labelnum=16  #for LA training
``` 

To test a model,
```
python ./code/test_LA.py --exp=OMF_lab8 --labelnum=8 #for LA testing
python ./code/test_LA.py --exp=OMF_lab16 --labelnum=16 #for LA testing
```
## Results
<img width="654" alt="截屏2024-01-26 16 45 52" src="https://github.com/vigilliu/OMF/assets/129838909/5bf7713e-3bb2-4064-8800-397e126246e8">
<img width="890" alt="截屏2024-01-26 16 46 19" src="https://github.com/vigilliu/OMF/assets/129838909/855c96e6-1d1a-47e2-998b-1eb85c0373af">

## Acknowledgements
Our code is largely based on [BCP:Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation]. Thanks for these authors for their valuable work, hope our work can also contribute to related research.

## Questions
If you have any questions, welcome contact me at '.edu.cn'



