import torch
import PIL 
PIL.PILLOW_VERSION = PIL.__version__
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from const import PAD, BOS
from PIL import Image
import pickle
import numpy as np
import glob
import json
from vocab_build import Vocabulary
import nltk

# print(WORD)

class Data_loader(object):
    def __init__(self, imgs, labels, max_len, batch_size, is_cuda, img_size=299, evaluation=False):
        self._imgs = imgs
        self._labels = np.asarray(labels)
        self._max_len = max_len
        self._is_cuda = is_cuda
        self.evaluation = evaluation
        self._step = 0
        self._batch_size = batch_size
        self.sents_size = len(imgs)
        self._stop_step = self.sents_size // batch_size
        self.names = None
        self._encode = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    def __iter__(self):
        return self

    def __next__(self):
        def img2variable(img_files):
            self.names = img_files
            tensors = [self._encode(Image.open(img_name).convert('RGB')).unsqueeze(0) for img_name in img_files]
            v = Variable(torch.cat(tensors, 0), volatile=self.evaluation)
            if self._is_cuda:
                v = v.cuda()
            return v

        def label2variable(labels):
            """maybe sth change between Pytorch versions, add func long() for compatibility
            """

            _labels = np.array(
                [l + [PAD] * (self._max_len - len(l)) for l in labels])

            _labels = Variable(torch.from_numpy(_labels),
                               volatile=self.evaluation).long()
            if self._is_cuda:
                _labels = _labels.cuda()
            return _labels

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        self._step += 1


        _imgs = img2variable(self._imgs[_start:_start + self._batch_size])
        _labels = label2variable(
            self._labels[_start:_start + self._batch_size])
        return _imgs, _labels

def get_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


if __name__ == "__main__":
    images = glob.glob( "C:\\Users\\Ruchika\\Downloads\\Compressed\\UCMerced_LandUse\\UCMerced_LandUse\\Images" + "\\*\\*")
    data = json.loads(open("C:\\Users\\Ruchika\\Downloads\\dataset_UCM.json", "r").read())
    captions = []
    vocab = get_vocab('E:\\ADC UCM\\vocab.pkl')
    for i in range(0, len(data['images'])):
        sentence = data['images'][i]['sentences'][0]['raw'] 
        tokens = nltk.tokenize.word_tokenize(str(sentence))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token.lower()) for token in tokens])
        caption.append(vocab('<end>'))
        captions.append(caption)

    training_data = Data_loader(images, captions, 20, batch_size=2, is_cuda=True)
    print(training_data.sents_size)
    img, labels = next(training_data)