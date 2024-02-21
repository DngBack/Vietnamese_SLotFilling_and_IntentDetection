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

def add_paddings(seq_out, MAX_LEN):
    sout=[]
    for i in range(len(seq_out)):
        # add padding inside output tokens
        temp = seq_out[i]
        if len(temp)<MAX_LEN:
            while len(temp)<MAX_LEN:
                temp.append('<PAD>')
        else:
            temp = temp[:MAX_LEN]
        sout.append(temp)
    return sout

def get_subtoken_mask(current_tokens, bert_tokenizer, MAX_LEN):
    '''
    Description:
        Create attention masks for BERT-based models that consider both word-level and subtoken-level information.
    Args:
        current_tokens: A list of input text strings.
        bert_tokenizer: A BERT tokenizer object used to split text into subtokens.
        MAX_LEN: An integer representing the maximum length of the input sequences.
    Returns:
        sub_mask: tensor sub-mask of sentences for BERT-based models.
    '''
    temp_mask = []
    for i in current_tokens:
        temp_row_mask = []
        temp_row_mask.append(False)  # for cls token
        temp = bert_tokenizer.tokenize(i)
        for j in temp:
            temp_row_mask.append(j[:2] != "##")  # Check if subtoken is not a padding token
        while len(temp_row_mask) < MAX_LEN:
            temp_row_mask.append(False)  # Pad mask to maximum length
        temp_mask.append(temp_row_mask)
        # if sum(temp_row_mask) != len(i.split(" ")):
        #     print(f"inconsistent:{temp}")
        #     print(i)
        #     print(sum(temp_row_mask))
        #     print(len(i.split(" ")))
    return torch.tensor(temp_mask).cuda()

# this function turns class text to id
def prepare_intent(intent, to_ix):
    '''
    Converts an intent text string to its corresponding integer ID.

    Args:
        intent (str): The text of the intent class.
        to_ix (dict): A dictionary mapping intent text strings to their integer IDs.

    Returns:
        list: The integer ID of the intent, or the ID of the "UNKNOWN" intent if the
             provided intent is not found in the dictionary.

    Raises:
        KeyError: If the provided intent is not found in the dictionary and there is no
                 "UNKNOWN" intent defined in the dictionary.
    '''
    idxs = to_ix[intent] if intent in to_ix.keys() else to_ix['<UNK>']
    return idxs

#this function converts tokens to ids and then to a tensor
def prepare_sequence(seq, to_ix):
    '''
    Converts a sequence of tokens to a PyTorch tensor of integer IDs.

    Args:
        seq (list): A list of tokens (words).
        to_ix (dict): A dictionary mapping tokens to their integer IDs.

    Returns:
        torch.Tensor: A PyTorch tensor of integer IDs, where each element corresponds
                      to the ID of the corresponding token in the original sequence.

    Raises:
        KeyError: If any token in the sequence is not found in the `to_ix` dictionary.

    '''
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix['<UNK>'], seq))
    return idxs

# converts numbers to <NUM> TAG
def number_to_tag(txt):
    return "<NUM>" if txt.isdecimal() else txt

# Here we remove multiple spaces and punctuation which cause errors in tokenization for bert & elmo.
def remove_punc(mlist):
    mlist = [re.sub(" +"," ",t.split("\t")[0][4:-4]) for t in mlist] # remove spaces down to 1
    temp_train_tokens = []
    # punct remove example:  play samuel-el jackson from 2009 - 2010 > play samuelel jackson from 2009 - 2010
    for row in mlist:
        tokens = row.split(" ")
        newtokens = []
        for token in tokens:
            newtoken = re.sub(r"[.,'\"\\/\-:&’—=–官方杂志¡…“”~%]",r"",token) # remove punc
            newtoken = re.sub(r"[楽園追放�]",r"A",newtoken)
            newtokens.append(newtoken if len(token)>1 else token)
        if newtokens[-1]=="":
            newtokens.pop(-1)
        if newtokens[0]=="":
            newtokens.pop(0)
        temp_train_tokens.append(" ".join(newtokens))
    return temp_train_tokens

def file2list(path):
    '''
    Get a list of text strings from a file.

    Args:
        path (str): The path to the file.

    Returns:
        list: A list of text strings.
    '''
    dataList = []

    with open(path, 'r') as f_r:
        data = f_r.readlines()
        for text in data:
            text = text.strip()
            dataList.append(text)

    return dataList

#defining datasets.
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def mask_important_tags(predictions,tags,masks):
    result_tags=[]
    result_preds=[]
    for pred,tag,mask in zip(predictions.tolist(),tags.tolist(),masks.tolist()):
        #index [0] is to get the data
        for p,t,m in zip(pred,tag,mask):
            if not m:
                result_tags.append(p)
                result_preds.append(t)
        #result_tags.pop()
        #result_preds.pop()
    return result_preds,result_tags

