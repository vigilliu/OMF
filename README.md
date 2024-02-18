# OMF:Overlay Mantle-Free for Semi-Supervised Medical Image Segmentation
by **************************************.
<img width="1483" alt="截屏2024-01-26 16 51 23" src="https://github.com/vigilliu/OMF/assets/129838909/9eacab39-29ae-43af-b0e9-c03b21de8c76">

## Introduction
Official code for "[Overlay Mantle-Free for Semi-Supervised Medical Image Segmentation](https://arxiv.org/)".

## Overview of our method
<img width="1053" alt="截屏2024-01-27 23 18 13" src="https://github.com/vigilliu/OMF/assets/129838909/bb78ecb5-4635-4dc8-9aab-615cf1493677">

## Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.6.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

## Usage
Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data).

To train a model,
```
python ./code/LA_BCP_train.py --exp=OMF_lab8 --base_lr=0.1 --labelnum=8  #for LA training 8:72
python ./code/LA_BCP_train.py --exp=OMF_lab16 --base_lr=0.1 --labelnum=16  #for LA training 16:64
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



