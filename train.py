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
from losses import *

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

#defining datasets.
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

class NLUDataset(Dataset):
    def __init__(self, sin,sout,intent,input_ids,attention_mask,token_type_ids,subtoken_mask, USE_CUDA):
        self.test = sin
        self.sin = Variable(torch.LongTensor(sin)).cuda() if USE_CUDA else Variable(torch.LongTensor(sin))
        self.sout = Variable(torch.LongTensor(sout)).cuda() if USE_CUDA else Variable(torch.LongTensor(sin))
        self.intent = Variable(torch.LongTensor(intent)).cuda() if USE_CUDA else Variable(torch.LongTensor(sin))
        self.input_ids=input_ids.cuda()
        self.attention_mask=attention_mask.cuda()
        self.token_type_ids=token_type_ids.cuda()
        self.subtoken_mask=subtoken_mask.cuda()
        self.x_mask = [Variable(torch.BoolTensor(tuple(map(lambda s: s ==1, t )))).cuda() for t in self.sin]
    def __len__(self):
        return len(self.intent)
    def __getitem__(self, idx):
        sample = self.sin[idx],self.sout[idx],self.intent[idx],self.input_ids[idx],self.attention_mask[idx],self.token_type_ids[idx],self.subtoken_mask[idx],self.x_mask[idx]
        return sample

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


def main(args):
    ENV_SEED=1331
    torch.manual_seed(ENV_SEED)
    random.seed(ENV_SEED)

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

    # # Dataloader 
    # #making single list
    train_data=NLUDataset(train_num_text, train_num_slotTag, train_num_label, dataset_toks['input_ids'], dataset_toks['attention_mask'], dataset_toks['token_type_ids'],train_subtoken_mask, USE_CUDA=USE_CUDA)
    test_data=NLUDataset(dev_num_text, dev_num_slotTag, dev_num_label, dev_toks['input_ids'], dev_toks['attention_mask'], dev_toks['token_type_ids'],dev_subtoken_mask, USE_CUDA= USE_CUDA)
    
    # print(train_data.__getitem__(4))
    # print("---------------------")
    # print(test_data.__getitem__(4))

    train_data = DataLoader(train_data, batch_size=args.BATCH_SIZE, shuffle=True)
    test_data = DataLoader(test_data, batch_size=args.BATCH_SIZE, shuffle=True)

    # Model setup 
    bert_layer = BertLayer()
    encoder = Encoder(len(word2index))
    middle = Middle()
    decoder = Decoder(len(tag2index),len(label2index))
    
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        middle = middle.cuda()
        bert_layer.cuda()


    dec_optim = optim.AdamW(decoder.parameters(),lr=0.0001)
    enc_optim = optim.AdamW(encoder.parameters(),lr=0.001)
    ber_optim = optim.AdamW(bert_layer.parameters(),lr=0.0001)
    mid_optim = optim.AdamW(middle.parameters(), lr=0.0001)
    enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, 1, gamma=0.96)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, 1, gamma=0.96)
    mid_scheduler = torch.optim.lr_scheduler.StepLR(mid_optim, 1, gamma=0.96)
    ber_scheduler = torch.optim.lr_scheduler.StepLR(ber_optim, 1, gamma=0.96)

    max_id_prec=0.
    max_sf_f1=0.
    max_id_prec_both=0.
    max_sf_f1_both=0.

    for step in tqdm(range(args.STEP_SIZE)):
        losses=[]
        id_precision=[]
        sf_f1=[]

        ### TRAIN
        encoder.train() # set to train mode
        middle.train()
        decoder.train()
        bert_layer.train()
        for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(train_data):
            batch_size=tag_target.size(0)
            bert_layer.zero_grad()
            encoder.zero_grad()
            middle.zero_grad()
            decoder.zero_grad()
            bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
            encoder_output = encoder(bert_last_hidden=bert_hidden)
            output = middle(encoder_output,bert_mask==0,training=True)
            start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
            start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)
            tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask, tag2index=tag2index)
            loss_1 = loss_function_1_smoothed(tag_score, tag_target.view(-1), num_classes=len(tag2index))
            loss_2 = loss_function_2_smoothed(intent_score,intent_target, num_classes=len(label2index))
            loss = loss_1+loss_2
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(middle.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(bert_layer.parameters(), 0.5)
            enc_optim.step()
            mid_optim.step()
            dec_optim.step()
            ber_optim.step()
            #print(bert_tokens[0])
            #print(tag_target[0])
            id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
            pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,args.MAX_LEN),tag_target,x_mask)
            sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))
        #print report
        print("Step",step," batches",i," :")
        print("Train-")
        print(f"loss:{round(float(np.mean(losses)),4)}")
        print(f"SlotFilling F1:{round(float(np.mean(sf_f1)),3)}")
        print(f"IntentDet Prec:{round(float(np.mean(id_precision)),3)}")
        losses=[]
        sf_f1=[]
        id_precision=[]

        #### Eval 
        encoder.train() # set to train mode
        middle.train()
        decoder.train()
        bert_layer.train()
         # to turn off gradients computation
        for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(test_data):
            batch_size=tag_target.size(0)
            encoder.zero_grad()
            middle.zero_grad()
            decoder.zero_grad()
            bert_layer.zero_grad()
            bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
            encoder_output = encoder(bert_last_hidden=bert_hidden)
            output = middle(encoder_output,bert_mask==0,training=True)
            start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
            start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)
            tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,tag2index=tag2index, infer=True)
            loss_1 = loss_function_1_smoothed(tag_score, tag_target.view(-1), num_classes=len(tag2index))
            loss_2 = loss_function_2_smoothed(intent_score,intent_target, num_classes=len(label2index))
            loss = loss_1 + loss_2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(middle.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(bert_layer.parameters(), 0.5)
            enc_optim.step()
            mid_optim.step()
            dec_optim.step()
            ber_optim.step()
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
            id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
            pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,args.MAX_LEN),tag_target,x_mask)
            sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))
        print("Test-")
        print(f"loss:{round(float(np.mean(losses)),4)}")
        print(f"SlotFilling F1:{round(float(np.mean(sf_f1)),4)}")
        print(f"IntentDet Prec:{round(float(np.mean(id_precision)),4)}")
        print("--------------")
        max_sf_f1 = max_sf_f1 if round(float(np.mean(sf_f1)),4)<=max_sf_f1 else round(float(np.mean(sf_f1)),4)
        max_id_prec = max_id_prec if round(float(np.mean(id_precision)),4)<=max_id_prec else round(float(np.mean(id_precision)),4)
        if max_sf_f1_both<=round(float(np.mean(sf_f1)),4) and max_id_prec_both<=round(float(np.mean(id_precision)),4):
            max_sf_f1_both=round(float(np.mean(sf_f1)),4)
            max_id_prec_both=round(float(np.mean(id_precision)),4)
            torch.save(bert_layer,f"models/ctran{args.fn}-bertlayer.pkl")
            torch.save(encoder,f"models/ctran{args.fn}-encoder.pkl")
            torch.save(middle,f"models/ctran{args.fn}-middle.pkl")
            torch.save(decoder,f"models/ctran{args.fn}-decoder.pkl")
        enc_scheduler.step()
        dec_scheduler.step()
        mid_scheduler.step()
        ber_scheduler.step()
    print(f"max single SF F1: {max_sf_f1}")
    print(f"max single ID PR: {max_id_prec}")
    print(f"max mutual SF:{max_sf_f1_both}  PR: {max_id_prec_both}")

if __name__ == "__main__":
    main(args)


