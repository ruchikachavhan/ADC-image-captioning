# ADC-image-captioning
Code for our paper: [A Novel Actor Dual-Critic Model for Image Captioning](https://arxiv.org/abs/2010.01999), ICPR 2020 <br>
[Ruchika Chavhan](https://ruchikachavhan.github.io/), [Biplab Banerjee](https://biplab-banerjee.github.io/), [Xiao Xiang Zhu](https://www.lrg.tum.de/sipeo/home/), [Subhasis Chaudhuri](https://www.ee.iitb.ac.in/~sc/)

# Datasets
1) [Remote Sensing Image Captioning Dataset (RSICD)](https://github.com/201528014227051/RSICD_optimal) <br>
2) [UC-Merced Captioning Dataset](http://vision.ucmerced.edu/datasets/) <br> 
Please make sure your data contains images and a json file containing captions corresponding to each image. 

# Methodology
<img width="651" alt="for_git" src="https://user-images.githubusercontent.com/32021556/106123090-7ebd2d00-617f-11eb-9908-5ccfb69ed1a1.PNG">

# Training

## Step 1: Build vocabulary

```
python vocab_build.py --caption_path <path to json file> --vocab_path <path to save the vocab file>
```
  
## Step 2: Train Model

### To load a pretrained Actor and Value Network model and train 

```
python train.py --data_path <path to images> --json_path <path to json file> --load_pretrained --actor_pretrained <path to actor model> --critic_pretrained <path to critic model>
```
  
### To train from scratch
```
python train.py --data_path <path to images> --json_path <path to json file>
```
  
## Citation
If you find this work useful, please cite our paper, 
```
@inproceedings{chavhan2020novel,
author    = {Ruchika Chavhan and Biplab Banerjee and Xiao Xiang Zhu and Subhasis Chaudhuri},
title     = {A Novel Actor Dual-Critic Model for Remote Sensing Image Captioning},
booktitle = {International Conference on Pattern Recognition (ICPR)},
year      = {2020}
}
```
<br> 

A significant part of this code has been adapted from the code from [Actor-Critic Sequence Training for Image Captioning](https://arxiv.org/abs/1706.09601)
