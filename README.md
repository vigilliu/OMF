# OMF:Overlay Mantle-Free for Semi-Supervised Medical Image Segmentation
by **************************************.
<img width="1483" alt="fig1111" src="https://github.com/vigilliu/OMF/assets/129838909/d5cc0555-122c-4f19-8579-f3a3f1049164">


## Introduction
Official code for "[Overlay Mantle-Free for Semi-Supervised Medical Image Segmentation]

## Overview of our method
<img width="1056" alt="fig111" src="https://github.com/vigilliu/OMF/assets/129838909/a8a466fa-4352-452a-808e-2578040c79a7">


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
<img width="654" alt="table111" src="https://github.com/vigilliu/OMF/assets/129838909/6141c801-f8d3-4ed0-8035-e221baf24dda">

<img width="890" alt="fig222" src="https://github.com/vigilliu/OMF/assets/129838909/122911df-3f5f-42f7-8804-080408a9f64d">

<img width="841" alt="fig333" src="https://github.com/vigilliu/OMF/assets/129838909/e7bf964f-740e-4229-bf33-9ee96e44ad00">

## Acknowledgements
Our code is largely based on [BCP:Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation]. Thanks for these authors for their valuable work, hope our work can also contribute to related research.

## Questions
If you have any questions,***********



