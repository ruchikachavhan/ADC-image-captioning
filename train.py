import argparse

parser = argparse.ArgumentParser(description='Actor Dual-Critic Image Cationing for Remote Sensing Data')
parser.add_argument('--logdir', type=str, default='tb_logdir')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--unuse_cuda', action='store_true')

parser.add_argument('--path', type=str, default='data/')
parser.add_argument('--json_path', type=str, default='data/data.json')
parser.add_argument('--save', type=str, default='imgcapt_v2_{}.pt')

parser.add_argument('--actor_pretrained', type=str, default='actor.pth')
parser.add_argument('--critic_pre_trained', type=str, default='critic.pth')

parser.add_argument('--actor_path', type=str, default='actor.pth')
parser.add_argument('--critic_path', type=str, default='critic.pth')
parser.add_argument('--enc_dec_path', type=str, default='actor.pth')

parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--new_lr', type=float, default=5e-6)
parser.add_argument('--load_pretrain', type=bool, default=True)
parser.add_argument('--actor_epochs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--iterations', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dec_hsz', type=int, default=256)
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--grad_clip', type=float, default=1.)

args = parser.parse_args()

import pickle
from vocab_build import Vocabulary
import glob 
import json
import nltk
from nltk import ngrams
from nltk.translate.bleu_score import modified_precision
import numpy as np
from utils import *


import torch

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda

if use_cuda:
    torch.cuda.manual_seed(args.seed)



# ##############################################################################
# Load datasets, While creating a function to load dataset in a different file, pickle shows an error. So to avoid that do this in train.py itself.
################################################################################
from create_dataset import Data_loader

vocab = get_vocab('vocab.pkl')
args.vocab_size = len(vocab)
args.max_len = 30

images = glob.glob(args.path+ "/*") 
data = json.loads(open(args.json_path, "r").read())['images']
captions = []
dataImages = []
print(images)
vocab = get_vocab('vocab.pkl')
for i in range(0, len(data)):
    sentence = data[i]['sentences'][0]['raw'] 
    tokens = nltk.tokenize.word_tokenize(str(sentence))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token.lower()) for token in tokens])
    caption.append(vocab('<end>'))
    if(len(caption) <= 30):
    	captions.append(caption)
    	dataImages.append(images[i])

training_data = Data_loader(dataImages, captions, args.max_len, batch_size=args.batch_size, is_cuda=True)
print("Dataset Loaded !")

# ##############################################################################
# Build model
# ##############################################################################
import models
from const import PAD
from optim import Optim, Policy_optim

actor = models.Actor(args.vocab_size,
                    args.dec_hsz,
                    args.rnn_layers,
                    args.batch_size,
                    args.max_len,
                    args.dropout,
                    use_cuda)

critic = models.Critic(args.vocab_size,
                      args.dec_hsz,
                      args.rnn_layers,
                      args.batch_size,
                      args.max_len,
                      args.dropout,
                      use_cuda)

EncoderDecoder = models.EncDec(args.dec_hsz, args.vocab_size).cuda()

if(args.load_pretrained):
    actor = load_checkpoint(actor, args.actor_pretrained)
    critic = load_checkpoint(critic, args.critic_pre_trained)


optim_pre_A = Optim(actor.get_trainable_parameters(),
                    args.lr, True, args.grad_clip)
optim_pre_C = Optim(critic.parameters(), args.lr, True,
                    args.grad_clip, weight_decay=0.5)
optim_pre_ED = torch.optim.RMSprop(EncoderDecoder.parameters(), lr=0.0005)


optim_A = Policy_optim(actor.get_trainable_parameters(), args.lr,
                       args.new_lr, args.grad_clip)
optim_C = Optim(critic.parameters(), args.lr,
                False, args.new_lr, args.grad_clip)

optim_ED = torch.optim.RMSprop(EncoderDecoder.parameters(), lr=0.0005)

criterion_A = torch.nn.CrossEntropyLoss(ignore_index=PAD)
criterion_C = torch.nn.MSELoss()

if use_cuda:
    actor = actor.cuda()
    critic = critic.cuda()

# ##############################################################################
# Training
# ##############################################################################
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F

from rouge import rouge_l, mask_score


def pre_train_actor(epoch):
    index = 0
    b1, b2, b3, b4, rouge = 0.0,0.0,0.0,0.0, 0.0
    for imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Pre-train Actor",
                             leave=False):
        optim_pre_A.zero_grad()
        actor.zero_grad()

        index+=1
        enc = actor.encode(imgs)
        hidden = actor.feed_enc(enc)
        target = actor(hidden, labels)
        _, words = actor(hidden)
        loss = criterion_A(target.view(-1, target.size(2)), labels.view(-1))
        loss.backward()
        optim_pre_A.step()

def pre_train_critic():
    iterations = 0
    actor.eval()
    critic.train()
    for imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Pre-train Critic",
                             leave=False):
        optim_pre_C.zero_grad()
        critic.zero_grad()
        enc = actor.encode(imgs)
        hidden_A = actor.feed_enc(enc)
        # we pre-train the critic network by feeding it with sampled actions from the fixed pre-trained actor.
        _, words = actor(hidden_A)
        policy_values = rouge_l(words, labels)

        hidden_C = critic.feed_enc(enc)
        estimated_values = critic(words, hidden_C)
        loss = criterion_C(estimated_values, policy_values)
        loss.backward()
        optim_pre_C.clip_grad_norm()
        optim_pre_C.step()

        iterations += 1
   
    
def pre_train_enndec():
    actor.eval()
    b1, b2, b3, b4, rouge = 0.0,0.0,0.0,0.0, 0.0
    for imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Pre-train EncoderDecoder",
                             leave=False):
        optim_pre_ED.zero_grad()
        EncoderDecoder.zero_grad()
        enc = actor.encode(imgs)
        # we pre-train the critic network by feeding it with sampled actions from the fixed pre-trained actor.
        loss, acc = EncoderDecoder(enc, labels)

        loss.backward()
        optim_pre_ED.step()
        


def train_actor_critic(GAMMA, epoch):
    actor.train()
    critic.train()
    EncoderDecoder.train()
    index = 0
    b1, b2, b3, b4, rouge = 0.0,0.0,0.0,0.0, 0.0
    for imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Training",
                             leave=False):
        optim_A.zero_grad()
        optim_C.zero_grad()
        EncoderDecoder.zero_grad()
        index+=1
        enc = actor.encode(imgs)
        hidden_A = actor.feed_enc(enc)
        target, words = actor(hidden_A)
        policy_values = rouge_l(words, labels)

        WriteInfiles(words, training_data.names, epoch, vocab, labels)
        hidden_C = critic.feed_enc(enc)
        estimated_values = critic(words, hidden_C)

        loss_c = criterion_C(estimated_values, policy_values)
        loss_c.backward()
        optim_C.clip_grad_norm()
        optim_C.step()

        reward = torch.mean(policy_values - estimated_values)

        loss_a = criterion_A(target.view(-1, target.size(2)), labels.view(-1))
        loss_a.backward()
        optim_A.clip_grad_norm()
        optim_A.step(reward)

        actor.zero_grad()
        EncoderDecoder.zero_grad()
        enc = actor.encode(imgs)
        hidden_A = actor.feed_enc(enc)
        target, words = actor(hidden_A)
        lossGen, accGen = EncoderDecoder(enc, words)
        lossReal, accReal = EncoderDecoder(enc, labels)
        loss_a = criterion_A(target.view(-1, target.size(2)), labels.view(-1))
        loss_a.backward()
        optim_A.clip_grad_norm()
        
        A = accReal - GAMMA*accGen
        A = A.view(-1)
        optim_A.step(A)
        EncoderDecoder.zero_grad()
        lossReal.backward()
        optim_ED.step()

        WriteInfiles(words, training_data.names, epoch, vocab, labels)
        bleuVals = getScores(words, labels)
        b1 += bleuVals[0]
        b2 += bleuVals[1]
        b3 += bleuVals[2]
        b4 += bleuVals[3]
        rouge += torch.mean(policy_values).item()
        print("LOSS", loss_a.item())

    b1, b2, b2, b4, rouge =  (b1/index, b2/index, b2/index, b4/index, rouge/index)
    print("AVERAGE SCORES B1:, B2:, B3:, B4:, ROUGE:", b1, b2, b2, b4, rouge)


def eval():


    actor.eval()
    b1, b2, b3, b4 = 0.0, 0.0, 0.0, 0.0
    rouge = 0.0
    index = 0
    for imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Eval",
                             leave=False):
        index += 1
        enc = actor.encode(imgs)

        hidden = actor.feed_enc(enc)
        target, words = actor(hidden)
        
        words = correctSentence(words)
        words = torch.tensor(words)
        words = words.unsqueeze(0)
        labels = correctSentence(labels)
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)
        policy_values = rouge_l(words, labels)
        rouge+= policy_values
        b1+= getScores(words, labels)[0]
        b2+= getScores(words, labels)[1]
        b3+= getScores(words, labels)[2]
        b4+= getScores(words, labels)[3]

    b1, b2, b2, b4, rouge =  (b1/index, b2/index, b2/index, b4/index, rouge/index)
    print("AVERAGE SCORES B1:, B2:, B3:, B4:, ROUGE:", b1, b2, b2, b4, rouge)
       
# Writes predicted sentences in a json file
def testSentences():
    actor.eval()
    listPred = []
    listTrue = []
    for imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Eval",
                             leave=False):
        enc = actor.encode(imgs)

        hidden = actor.feed_enc(enc)
        target, words = actor(hidden)
        
        words = correctSentence(words)
        words = torch.tensor(words)
        words = words.unsqueeze(0)
        labels = correctSentence(labels)
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)

        predicted = getSentence(vocab, words)
        truth = getSentence(vocab, labels)

        dictPred = {"image_id": training_data.names[0].split("\\")[-1], "caption": predicted}
        dictTrue = {"image_id": training_data.names[0].split("\\")[-1], "caption": truth}

        listPred.append(dictPred)
        listTrue.append(dictTrue)

    with open('predicted.json', 'w') as fout:
        json.dump(listPred , fout)
    with open('ground_truth.json', 'w') as fout:
        json.dump(listTrue , fout)



try:
    if(not(args.load_pretrained)):
        actor.train()
        for step in range(15):
            pre_train_actor(step)
            model_state_dict = actor.state_dict()
            model_source = {
                "model": model_state_dict,
            }
            torch.save(model_source, args.actor_path)

        for step in range(0, 10):
            pre_train_critic()
            model_state_dict = critic.state_dict()
            model_source = {
                "model": model_state_dict,
            }
            torch.save(model_source, args.critic_path)


    EncoderDecoder.train()
    for step in range(0, 10):
        pre_train_enndec()
        model_state_dict = EncoderDecoder.state_dict()
        model_source = {
            "model": model_state_dict,
        }
        torch.save(model_source, args.enc_dec_path)

    GAMMA = 0.01
    for step in range(args.epochs):
        train_actor_critic(GAMMA, step)
        GAMMA += 0.01
        model_state_dict = actor.state_dict()
        model_source = {
            "model": model_state_dict,
        }
        torch.save(model_source, args.actor_path)
        model_state_dict = critic.state_dict()
        model_source = {
            "model": model_state_dict,
        }
    eval()
      

except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early")
