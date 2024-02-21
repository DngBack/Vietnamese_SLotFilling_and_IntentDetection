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

# Get model and utils
from models import * 
from utils import *
from preprocess import * 

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
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--train_input_path', type=str, default="./dataset/bkai_dataset/training_data/training_data/seq.in", help="The text input train path")
    parser.add_argument('--train_labels_path', type=str, default="./dataset/bkai_dataset/training_data/training_data/label", help="The label of train")
    parser.add_argument('--train_intentTag_path', type=str, default="./dataset/bkai_dataset/training_data/training_data/seq.out", help="The intent Tag labels input train path")
    parser.add_argument('--dev_input_path', type=str, default="./dataset/bkai_dataset/dev_data/dev_data/seq.in")
    parser.add_argument('--dev_labels_path', type=str, default="./dataset/bkai_dataset/dev_data/dev_data/label")
    parser.add_argument('--dev_slotTag_path', type=str, default="./dataset/bkai_dataset/dev_data/dev_data/seq.out")
    parser.add_argument('--fn', type=str, default="final", help="file unique id for saving and loading models")
    parser.add_argument('--MAX_LEN', type=int, default=128, help="Length of tokens input BERT")
    parser.add_argument('--ENV_BERT_ID_CLS', type= bool, default=False, help="use cls token for id classification")
    parser.add_argument('--ENV_EMBEDDING_SIZE', type= int, default=768, help="dimention of embbeding, bertbase=768,bertlarge&elmo=1024")
    parser.add_argument('--ENV_CNN_FILTERS', type= int, default=128)
    parser.add_argument('--ENV_CNN_KERNELS', type= int, default=4)
    parser.add_argument('--BATCH_SIZE', type= int, default=32)
    parser.add_argument('--STEP_SIZE', type= int, default=10)

    args = parser.parse_args()
    return args


# args = getConfig_Input()
args = parse_args()

def inference(args):
    ENV_SEED=1331
    # you must use cuda to run this code.
    USE_CUDA = torch.cuda.is_available()

    # Get data 
    # Get data from file
    train_text = file2list(args.train_input_path)
    train_label = file2list(args.train_labels_path)
    train_intentTag = file2list(args.train_intentTag_path)

    dev_text = file2list(args.dev_input_path)
    dev_label = file2list(args.dev_labels_path)
    dev_intentTag = file2list(args.dev_slotTag_path)

    # Text Preprocesing
    train_text = [normalize_text(text) for text in train_text]
    dev_text = [normalize_text(text) for text in dev_text]

    # Get all unique tokens from labels
    unique_labels = set(train_label)

    # Create dictionary token for labels
    #initialize intent to index
    label2index={'UNKNOWN':0}
    for label in unique_labels:
        if label not in label2index.keys():
            label2index[label] = len(label2index)

    # Covert from index to labels
    index2intent = {v:k for k,v in label2index.items()}

    # Get all unique tokens from intentTag
    intentTag = []
    for tag in train_intentTag:
        intentTag.extend(tag.split())

    unique_intentTag = set(intentTag)

    # Create a dictionảy token for intentTag
    # Tag dictionary
    tag2index = {'<BOS>':0, '<PAD>' : 1, '<EOS>':2, '<UNK>':3}

    for tag in unique_intentTag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # Covert from index to tag
    index2tag = {v:k for k,v in tag2index.items()}

    # Khởi tạo dictionary cho input
    train_toks_text = []

    for sen_text in train_text:
        listSenText = sen_text.split()
        train_toks_text.append(listSenText)

    for test_text in dev_text: 
        devSenText = test_text.split()
        train_toks_text.append(devSenText)


    vocab = []
    for lstSen in train_toks_text: 
        vocab.extend(lstSen)

    vocab = set(vocab) 

    # making dictionary (token:id), initial value
    word2index = {'<PAD>': 1, '<UNK>':0,'<BOS>':2,'<EOS>':3,'<NUM>':4}
    # add rest of token list to dictionary
    for token in vocab:
        if token not in word2index.keys():
            word2index[token]=len(word2index)

    # Convert from label
    train_num_label = [prepare_intent(temp,label2index) for temp in train_label]
    dev_num_label = [prepare_intent(temp,label2index) for temp in dev_label]

    # Convert from content Tag
    # Convert from to list per tag
    lst_slotTag = []
    for tag in train_intentTag:
        lst_slotTag.append(tag.split())

    train_num_slotTag = []
    for sen_slotTag in lst_slotTag:
        sen_slotTag.extend(['<PAD>']*(args.MAX_LEN -len(sen_slotTag)))
        sen_slotTag = [prepare_intent(temp, tag2index) for temp in sen_slotTag]
        train_num_slotTag.append(sen_slotTag)

    # Do the same with dev
    dev_lst_slotTag = []
    for dev_tag in dev_intentTag:
        dev_lst_slotTag.append(dev_tag.split())

    dev_num_slotTag = []
    for slotTag in dev_lst_slotTag:
        slotTag.extend(['<PAD>']*(args.MAX_LEN-len(slotTag)))
        slotTag = [prepare_intent(temp, tag2index) for temp in slotTag]
        dev_num_slotTag.append(slotTag)

    # Convert the text
    train_lst_text = []
    for text in train_text:
        train_lst_text.append(text.split())

    train_num_text = []
    for trainText in train_lst_text:
        trainText.extend(['<PAD>']*(args.MAX_LEN - len(trainText)))
        trainText = [prepare_intent(temp, word2index) for temp in trainText]
        train_num_text.append(trainText)

    # Do the same with dev_test
    dev_lst_text = []
    for devText in dev_text:
        dev_lst_text.append(devText.split())

    dev_num_text = []
    for devText in dev_lst_text:
        devText.extend(['<PAD>']*(args.MAX_LEN - len(devText)))
        devText = [prepare_intent(temp, word2index) for temp in devText]
        dev_num_text.append(devText)

    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")  # Or PhoBert-large

    dataset_toks = phobert_tokenizer.batch_encode_plus(train_text,
                                                max_length=args.MAX_LEN ,
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                return_attention_mask=True,
                                                padding='max_length',
                                                truncation=True)

    # Do the same with dev dataset
    dev_toks = phobert_tokenizer.batch_encode_plus(dev_text,
                                                max_length=args.MAX_LEN ,
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                return_attention_mask=True,
                                                padding='max_length',
                                                truncation=True)
    
    # Get subtoken mask
    train_subtoken_mask = get_subtoken_mask(train_text,phobert_tokenizer, args.MAX_LEN)
    dev_subtoken_mask = get_subtoken_mask(dev_text,phobert_tokenizer, args.MAX_LEN)
    # Model setup 
    bert_layer = BertLayer()
    encoder = Encoder(len(word2index))
    middle = Middle()
    decoder = Decoder(len(tag2index),len(label2index))

    # This cell reloads the best model during training from hard-drive.
    bert_layer.load_state_dict(torch.load(f'./models/ctranfinal-bertlayer.pkl').state_dict())
    encoder.load_state_dict(torch.load(f'./models/ctranfinal-encoder.pkl').state_dict())
    middle.load_state_dict(torch.load(f'./models/ctranfinal-middle.pkl').state_dict())
    decoder.load_state_dict(torch.load(f'./models/ctranfinal-decoder.pkl').state_dict())
    if USE_CUDA:
        bert_layer = bert_layer.cuda()
        encoder = encoder.cuda()
        middle = middle.cuda()
        decoder = decoder.cuda()
        
    print("Example of model prediction on test dataset")
    encoder.eval()
    middle.eval()
    decoder.eval()
    bert_layer.eval()

    with torch.no_grad():
        index = random.choice(range(len(dev_text)))
        test_raw = dev_text[index]
        print(test_raw)

        bert_tokens = dev_toks['input_ids'][index].unsqueeze(0).cuda()
        print(bert_tokens)
        bert_mask = dev_toks['attention_mask'][index].unsqueeze(0).cuda()
        bert_toktype = dev_toks['token_type_ids'][index].unsqueeze(0).cuda()
        subtoken_mask = dev_subtoken_mask[index].unsqueeze(0).cuda()
        test_in = Variable(torch.LongTensor(prepare_sequence(test_raw,word2index))).cuda()
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
        start_decode = Variable(torch.LongTensor([[word2index['<BOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<BOS>']]*1])).transpose(1,0)
        # test_raw = [removepads(torch.LongTensor(test_raw))]
        bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
        encoder_output = encoder(bert_last_hidden=bert_hidden)
        output = middle(encoder_output,bert_mask==0)
        tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=True)

        v,i = torch.max(tag_score,1)
        print("Sentence           : ",test_raw)
        print("Tag Truth          : ", dev_intentTag[index])
        print("Tag Prediction     : ", *(list(map(lambda ii:index2tag[ii],i.data.tolist()))[:len(test_raw)]))
        v,i = torch.max(intent_score,1)
        print("Intent Truth       : ", dev_label[index])
        print("Intent Prediction  : ",index2intent[i.data.tolist()[0]])

if __name__ == "__main__":
    inference(args)