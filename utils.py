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

# To remove keywords like <PAD> <UNK> which have index 2 and 3
def correctSentence(sentence):
    correct = []
    for w in range(0, sentence.shape[1]):
        if(sentence[0][w]== 2 or sentence[0][w]==3):
            break
        else:
            correct.append(sentence[0][w])
    return correct

def get_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_checkpoint(model, filepath):
    model.load_state_dict(torch.load(filepath)['model'])
    model.eval()
    return model

def  get_word( vocab , predicted):
        word_id = predicted.item()
        word = vocab.idx2word[word_id]
        return word

def get_pred(predicted):
    pred = []
    a = []
    for i in range(0, predicted.shape[1]):
        a.append(int(predicted[0][i]))
    pred.append(a)
    return pred


def getSentence(vocab, predicted):
    sentence = []
    for bs in range(0, predicted.shape[0]):
        captions = []
        for i in range(0, predicted.shape[1]):
            word = get_word(vocab, predicted[bs][i])
            captions.append(word)
        sentence.append(captions)
    return sentence

def WriteInfiles(words, fileNames, epoch, vocab, labels):
    sentences = getSentence(vocab, words)
    labels = labels.unsqueeze(1)
    for b in range(0, len(fileNames)):
        sentence = sentences[b]
        if(epoch == 0):
                targetSentence = getSentence(vocab, labels[b])
                name = "resultsA2C/"+ fileNames[b].split(".")[0].split("/")[-1]+ '.txt'
                f = open(name, "w")
                f.write(str(targetSentence))
                f.write("\n")
                f.write(str(epoch))
                f.write("\n")
                f.write(str(sentence))
                f.write("\n")
        else:
            name =  "resultsA2C/"+ fileNames[b].split(".")[0].split("/")[-1]+ '.txt'
            f = open(name, "a+")
            f.write(str(epoch))
            f.write("\n")
            f.write(str(sentence))
            f.write("\n")


def getScores(predicted, captions):
	predicted = predicted.unsqueeze(1)
	captions = captions.unsqueeze(1)
	bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
	bs = predicted.shape[0]


	for b in range(0, predicted.shape[0]):
		predIndex = get_pred(predicted[b])[0]
		captionIndex = get_pred(captions[b])

		bleu1 += nltk.translate.bleu_score.sentence_bleu(captionIndex,predIndex , weights= (1, 0, 0, 0)) 
		bleu2 += nltk.translate.bleu_score.sentence_bleu(captionIndex,predIndex, weights= (0, 1, 0, 0)) 
		bleu3 += nltk.translate.bleu_score.sentence_bleu(captionIndex,predIndex, weights= (0, 0, 1, 0))
		bleu4 += nltk.translate.bleu_score.sentence_bleu(captionIndex,predIndex, weights= (0, 0, 0, 1))

	bleu1, bleu2, bleu3, bleu4 = bleu1/bs, bleu2/bs, bleu3/bs, bleu4/bs
	return (bleu1, bleu2, bleu3, bleu4)
