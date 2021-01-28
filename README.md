# ADC-image-captioning
Code for our paper: [A Novel Actor Dual-Critic Model for Image Captioning](https://arxiv.org/abs/2010.01999), ICPR 2020 <br>
[Ruchika Chavhan](https://ruchikachavhan.github.io/), [Biplab Banerjee](https://biplab-banerjee.github.io/), [Xiao Xiang Zhu](https://www.lrg.tum.de/sipeo/home/), [Subhasis Chaudhuri](https://www.ee.iitb.ac.in/~sc/)

# Pre-requisites
Please make sure your data contains images and a json file containing captions corresponding to each image. 

# Training

## Step 1: Build vocabulary

'''
python vocab_build.py --caption_path <path to json file> --vocab_path <path to save the vocab file>
'''
  
## Step 2: Train Model

### To load a pretrained Actor and Value Network model and train 

'''
python train.py --data_path <path to images> --json_path <path to json file> --load_pretrained --actor_pretrained <path to actor model> --critic_pretrained <path to critic model>
'''
  
### To train from scratch
'''
python train.py --data_path <path to images> --json_path <path to json file>
'''
  

