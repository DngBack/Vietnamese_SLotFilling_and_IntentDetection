# Import Package

# Torch Library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Metric from sklearn
from sklearn.metrics import f1_score,accuracy_score

# Transformer to get Bert model
from transformers import AutoModel, AutoTokenizer

# Other
import random
import numpy as np
import pickle
from tqdm import tqdm
import math
import re
import pandas as pd
import os
import time



def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
  """
  Generates an upper-triangular matrix of -inf, with zeros on diag.

  Args:
    sz: int, the size of the mask to generate.

  Returns:
    A torch.Tensor of size (sz, sz) with -inf on the upper triangle and 0 on the
    diagonal.
  """

  return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def generate_square_diagonal_mask(sz: int) -> torch.Tensor:
  """
  Generates a square matrix of size (sz, sz) with zeros on the diagonal and
  -inf on all other entries.

  Args:
    sz: Size of the mask to generate.

  Returns:
    A torch.Tensor of size (sz, sz) with 0 on the diagonal and -inf on other
    entries.
  """

  return torch.triu(torch.ones(sz, sz)-float('inf'), diagonal=1)+torch.tril(torch.ones(sz,sz)-float('inf'), diagonal=-1)


# positional embedding used in transformers
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Get positional Encoding

        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]

        Return: 
            x: Tensor, adding positional encoding with input word embedding 
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
#start of the shared encoder
class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.bert_model = AutoModel.from_pretrained("vinai/phobert-base")


    def forward(self, bert_info=None):
        (bert_tokens, bert_mask, bert_tok_typeid) = bert_info
        bert_encodings = self.bert_model(bert_tokens, bert_mask, bert_tok_typeid)
        bert_last_hidden = bert_encodings['last_hidden_state']
        bert_pooler_output = bert_encodings['pooler_output']
        return bert_last_hidden, bert_pooler_output



class Encoder(nn.Module):
    def __init__(self, p_dropout=0.5, ENV_CNN_FILTERS=128, ENV_CNN_KERNELS=4, ENV_EMBEDDING_SIZE=768):
        super(Encoder, self).__init__()
        self.filter_number = ENV_CNN_FILTERS
        self.kernel_number = ENV_CNN_KERNELS  # tedad size haye filter : 2,3,5 = 3
        self.embedding_size = ENV_EMBEDDING_SIZE
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(2,),
                               padding="same", padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")
        self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(5,),
                               padding="same", padding_mode="zeros")
        self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(1,),
                               padding="same", padding_mode="zeros")

    def forward(self, bert_last_hidden):
        trans_embedded = torch.transpose(bert_last_hidden, dim0=1, dim1=2)
        convolve1 = self.activation(self.conv1(trans_embedded))
        convolve2 = self.activation(self.conv2(trans_embedded))
        convolve3 = self.activation(self.conv3(trans_embedded))
        convolve4 = self.activation(self.conv4(trans_embedded))
        convolve1 = torch.transpose(convolve1, dim0=1, dim1=2)
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        convolve3 = torch.transpose(convolve3, dim0=1, dim1=2)
        convolve4 = torch.transpose(convolve4, dim0=1, dim1=2)
        output = torch.cat((convolve4, convolve1, convolve2, convolve3), dim=2)
        return output

    
#Middle
class Middle(nn.Module):
    def __init__(self ,p_dropout=0.5, ENV_CNN_FILTERS=128, ENV_CNN_KERNELS=4, MAX_LEN=128):
        super(Middle, self).__init__()
        self.ENV_HIDDEN_SIZE = ENV_CNN_FILTERS*ENV_CNN_KERNELS
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        #Transformer
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.pos_encoder = PositionalEncoding(self.ENV_HIDDEN_SIZE, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(self.ENV_HIDDEN_SIZE, nhead=2,batch_first=True, dim_feedforward=2048 ,activation="relu", dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers,enable_nested_tensor=False)
        self.transformer_mask = generate_square_subsequent_mask(MAX_LEN).cuda()

    def forward(self, fromencoder,input_masking,training=True):
        src = fromencoder * math.sqrt(self.ENV_HIDDEN_SIZE)
        src = self.pos_encoder(src)
        output = (self.transformer_encoder(src,src_key_padding_mask=input_masking)) # outputs probably
        return output
    
#start of the decoder
class Decoder(nn.Module):

    def __init__(self,slot_size,intent_size,dropout_p=0.5, MAX_LEN=128, ENV_EMBEDDING_SIZE=768, ENV_CNN_FILTERS=128, ENV_CNN_KERNELS=4):
        super(Decoder, self).__init__()
        self.ENV_HIDDEN_SIZE = ENV_CNN_FILTERS*ENV_CNN_KERNELS
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.dropout_p = dropout_p
        self.softmax= nn.Softmax(dim=1)
        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.ENV_HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.slot_trans = nn.Linear(self.ENV_HIDDEN_SIZE, self.slot_size)
        self.intent_out = nn.Linear(self.ENV_HIDDEN_SIZE,self.intent_size)
        self.intent_out_cls = nn.Linear(ENV_EMBEDDING_SIZE,self.intent_size) # dim of bert
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.ENV_HIDDEN_SIZE, nhead=2,batch_first=True,dim_feedforward=300 ,activation="relu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.transformer_mask = generate_square_subsequent_mask(MAX_LEN).cuda()
        self.transformer_diagonal_mask = generate_square_diagonal_mask(MAX_LEN).cuda()
        self.pos_encoder = PositionalEncoding(self.ENV_HIDDEN_SIZE, dropout=0.1)
        self.self_attention = nn.MultiheadAttention(embed_dim=self.ENV_HIDDEN_SIZE
                                                    ,num_heads=8,dropout=0.1
                                                    ,batch_first=True)
        self.layer_norm = nn.LayerNorm(self.ENV_HIDDEN_SIZE)


    def forward(self, input,encoder_outputs,encoder_maskings,bert_subtoken_maskings=None,infer=False, tag2index={}):
        # encoder outputs: BATCH,LENGTH,Dims (16,60,1024)
        batch_size = encoder_outputs.shape[0]
        length = encoder_outputs.size(1) #for every token in batches
        embedded = self.embedding(input)

        # print("NOT CLS")
        encoder_outputs2=encoder_outputs
        context,attn_weight = self.self_attention(encoder_outputs2,encoder_outputs2,encoder_outputs2
                                                  ,key_padding_mask=encoder_maskings)
        encoder_outputs2 = self.layer_norm(self.dropout2(context))+encoder_outputs2
        sum_mask = (~encoder_maskings).sum(1).unsqueeze(1)
        sum_encoder = ((((encoder_outputs2)))*((~encoder_maskings).unsqueeze(2))).sum(1)
        intent_score = self.intent_out(self.dropout1(sum_encoder/sum_mask)) # B,D


        newtensor = torch.cuda.FloatTensor(batch_size, length,self.ENV_HIDDEN_SIZE).fill_(0.) # size of newtensor same as original
        for i in range(batch_size): # per batch
            newtensor_index=0
            for j in range(length): # for each token
                if bert_subtoken_maskings[i][j].item()==1:
                    newtensor[i][newtensor_index] = encoder_outputs[i][j]
                    newtensor_index+=1

        if infer==False:
            embedded=embedded*math.sqrt(self.ENV_HIDDEN_SIZE)
            embedded = self.pos_encoder(embedded)
            zol = self.transformer_decoder(tgt=embedded,memory=newtensor
                                           ,memory_mask=self.transformer_diagonal_mask
                                           ,tgt_mask=self.transformer_mask)

            scores = self.slot_trans(self.dropout3(zol))
            slot_scores = F.log_softmax(scores,dim=2)
        else:
            bos = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
            bos = self.embedding(bos)
            tokens=bos
            for i in range(length):
                temp_embedded=tokens*math.sqrt(self.ENV_HIDDEN_SIZE)
                temp_embedded = self.pos_encoder(temp_embedded)
                zol = self.transformer_decoder(tgt=temp_embedded,
                                               memory=newtensor,
                                               tgt_mask=self.transformer_mask[:i+1,:i+1],
                                               memory_mask=self.transformer_diagonal_mask[:i+1,:]
                                               )
                scores = self.slot_trans(self.dropout3(zol))
                softmaxed = F.log_softmax(scores,dim=2)
                #the last token is apended to vectors
                _,input = torch.max(softmaxed,2)
                newtok = self.embedding(input)
                tokens=torch.cat((bos,newtok),dim=1)
            slot_scores = softmaxed

        return slot_scores.view(input.size(0)*length,-1), intent_score




