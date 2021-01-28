import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
import math
import numpy as np
from const import BOS, PAD

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)  #USing pretrained Alexnet
        self.linear = nn.Linear(self.alexnet.classifier[6].in_features, embed_size)
        self.alexnet.classifier[6] = self.linear
        self.norm1 = nn.LayerNorm(embed_size)
    def forward(self, images):
        """Extract feature vectors from input images.""" 
        f = self.alexnet(images)
        return (f, f)


class Actor(nn.Module):
    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()

        self.torch = torch.cuda if use_cuda else torch
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.enc = EncoderCNN(dec_hsz)
        self.enc_out = nn.Linear(dec_hsz, dec_hsz)
        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(dec_hsz + dec_hsz, dec_hsz, rnn_layers,
                           batch_first=True,
                           dropout=dropout)
        self.out = nn.Linear(self.dec_hsz, vocab_size)

        self._reset_parameters()

    def forward(self, hidden, labels=None):
        word = Variable(self.torch.LongTensor([[BOS]] * self.bsz))
        emb_enc = self.lookup_table(word)
        hiddens = [hidden[0].squeeze()]
        outputs, words = [], []

        h= hidden[0].view(hidden[0].shape[1], hidden[0].shape[0], hidden[0].shape[2])
        for i in range(self.max_len):
            _, hidden = self.rnn(torch.cat([emb_enc, h], -1), hidden)
            h_state = F.dropout(hidden[0], p=self.dropout)

            props = F.log_softmax(self.out(h_state[-1]), dim=-1)
            h = h= hidden[0].view(hidden[0].shape[1], hidden[0].shape[0], hidden[0].shape[2])
            if labels is not None:
                emb_enc = self.lookup_table(labels[:, i]).unsqueeze(1)

            else:
                _props = props.data.clone().exp()
                word = Variable(_props.multinomial(1), requires_grad=False)
                words.append(word)
                emb_enc = self.lookup_table(word)
            outputs.append(props.unsqueeze(1))

        if labels is not None:
            return torch.cat(outputs, 1)

        else:
            return torch.cat(outputs, 1), torch.cat(words, 1)

    def encode(self, imgs):
        enc = self.enc(imgs)[0]
        enc = self.enc_out(enc)
        return enc

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(
            self.rnn_layers, self.bsz, self.dec_hsz).zero_())

        h = Variable(enc.data.
                     unsqueeze(0).expand(self.rnn_layers, *enc.size()))

        return (h.contiguous(), c.contiguous())

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)

        self.enc_out.weight.data.uniform_(-stdv, stdv)
        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.out.weight.data.uniform_(-stdv, stdv)

        for p in self.enc.parameters():
            p.requires_grad = False

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())

class GRUCell(nn.Module):
    def __init__(self, dim, drop_prob):
        super().__init__()
        self.gru = nn.GRU(dim, dim, 1, batch_first=True, dropout=drop_prob)
        # self.initialize()
    def forward(self, features, hiddens):
        out, hiddens = self.gru(features, hiddens)
        return out, hiddens
    def initialize(self):
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)



class Critic(nn.Module):
    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()

        self.use_cuda = use_cuda
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(self.dec_hsz,
                           self.dec_hsz,
                           self.rnn_layers,
                           batch_first=True,
                           dropout=dropout)
        self.value = nn.Linear(self.dec_hsz, 1)

        self._reset_parameters()

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(
            self.rnn_layers, self.bsz, self.dec_hsz).zero_())

        h = Variable(enc.data.
                     unsqueeze(0).expand(self.rnn_layers, *enc.size()))
        return (h.contiguous(), c.contiguous())

    def forward(self, inputs, hidden):
        emb_enc = self.lookup_table(inputs.clone()[:, :-1])
        _, out = self.rnn(emb_enc, hidden)
        out = F.dropout(out[0][-1], p=self.dropout)

        return self.value(out).squeeze()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)

        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.value.weight.data.uniform_(-stdv, stdv)


class EncDec(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(EncDec, self).__init__()
        self.embed_size = embed_size
        self.enc = GRUCell(embed_size, 0.0)
        self.dec = GRUCell(embed_size, 0.0)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.requires_grad = True
        self.linear = nn.Linear(embed_size, embed_size)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.relu = nn.LeakyReLU(0.2, inplace = True)
        self.drop = nn.Dropout(0.5)


    def forward(self, features, gen_captions):
        cos = nn.CosineSimilarity(dim=1)
        L1Loss = nn.L1Loss()
        gru_hiddens = features.unsqueeze(0)
        Loss = 0.0
        acc = 0.0
        info = torch.zeros(gen_captions.shape[0], 1, self.embed_size).cuda()
        encoded = []
        for g in range(0, gen_captions.shape[1]):
            gen_cap = self.embed(gen_captions[:, g]).unsqueeze(1)
            gru_enc_out, gru_hiddens = self.enc(gen_cap, gru_hiddens)
            encoded.append(gru_enc_out)

        gru_hiddens_2 = self.relu(self.linear1(gru_hiddens))

        for g in range(0, gen_captions.shape[1]):
            input = self.relu(self.linear(encoded[g]))
            gru_dec_out, gru_hiddens_2 = self.dec(input, gru_hiddens_2)
            info += gru_dec_out

        info = info/gen_captions.shape[1]
        info = self.relu(self.linear2(info))

        Loss = EPE(info, features)
        # info = info.squeeze(1)
        features = features.unsqueeze(1)
        for b in range(0, gen_captions.shape[0]):
            f = F.normalize(features[b])
            info_ = F.normalize(info[b]).view(info[b].shape[1], 1)
            acc += torch.mm(f, info_)

        acc = acc/gen_captions.shape[0]

        return Loss, acc

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())