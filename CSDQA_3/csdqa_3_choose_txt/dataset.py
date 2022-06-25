from __future__ import print_function
import os
import sys
import json
import re
import _pickle as cPickle # python3
import numpy as np

import torch
from torch.utils.data import Dataset
from obj_info_embedding import Info_Embedding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

from transformers import AutoTokenizer
bert_models = {'bert-base-uncased': 'bert-base-uncased',
               'bert-tiny':   'google/bert_uncased_L-2_H-128_A-2',
               'bert-mini':   'google/bert_uncased_L-4_H-256_A-4',
               'bert-small':  'google/bert_uncased_L-4_H-512_A-8',
               'bert-medium': 'google/bert_uncased_L-8_H-512_A-8',
               'bert-base':   'google/bert_uncased_L-12_H-768_A-12'}

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('\nloading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word) # initialize the instance
        print('vocabulary number in the dictionary:', len(idx2word))
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _simplify_question(ques):
    """
    Simplify question: remove verbose sentences in the question.
    """
    sentences = ques.split(". ")
    if len(sentences) > 1 and "Count" in sentences[0] and " by " in sentences[0]:
        ques = ". ".join(sentences[1:])
        return ques
    else:
        return ques


def pid2img(problems, pid):
    """
    Return the image's name about the problem
    """
    data = problems[pid]
    pattern = re.compile(r"CSDQA_3/.*[1-9][0-9]*.png$")
    result = re.search(pattern, data["diagram_path"])
    return result.group()[8:]


def _load_dataset(dataroot, name):
    """
    Load the CCKS dataset.
    - dataroot: root path of dataset
    - name: 'train', 'val', 'test'
    """
    problems =  json.load(open(os.path.join(dataroot, 'problems.json')))
    pid_splits = json.load(open(os.path.join(dataroot, 'pid_splits.json')))

    pids = pid_splits['%s' % name]
    print("problem number for %s:" % name, len(pids))

    entries = []
    
    for pid in pids:
        prob = {}
        prob['question_id'] = pid
        prob['image_name'] = pid2img(problems, pid)
        prob['question'] = _simplify_question(problems[pid]['question'])

        if name in ['train', 'val']:
            # answer to label
            ans = str(problems[pid]['correct_answer'])
            prob['answer'] = ans
            prob['answer_label'] = int(ord(ans) - 97)

        prob['choices'] = problems[pid]['answer']

        entries.append(prob)


    return entries


class ccksFeatureDataset(Dataset):
    def __init__(self, name, feat_label, dataroot, dictionary, lang_model, max_length, obj_max_num):
        super(ccksFeatureDataset, self).__init__ ()
        assert name in ['train', 'val', 'test']
        assert 'bert' in lang_model

        self.dictionary = dictionary
        self.lang_model = lang_model
        self.max_length = max_length # max question word length
        self.c_num = 4 # max choice number
        self.max_choice_length = 8 # max choice word length
        self.name = name
        self.obj_dim = 128

        # load and tokenize the questions and choices
        self.entries = _load_dataset(dataroot, name)
        if 'bert' in self.lang_model:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_models[self.lang_model]) # For Bert
        self.tokenize()
        self.tensorize()

        self.info_embed = Info_Embedding(lang_model, dataroot, obj_max_num)

        self.img2index = json.load(open("/home/chenyang/code/CCKS2022/CSDQA_3/data/CSDQA_3/img2index.json"))

        # load image features
        h5_path = os.path.join(dataroot, 'patch_embeddings', feat_label, 'ccks_%s_%s.pth' % (name, feat_label))
        print('\nloading features from h5 file:', h5_path)
        self.features = torch.load(h5_path)
        self.v_dim = list(self.features.values())[0].size()[1] # [num_patch,2048]
        print("visual feature dim:", self.v_dim) 

    def tokenize(self):
        """
        Tokenize the questions.
        This will add q_token in each entry of the dataset.
        """
        print('max question token length is:', self.max_length)
        print('max choice token length is:', self.max_choice_length)

        for entry in self.entries:
            if 'bert' in self.lang_model: # For Bert
                tokens = self.tokenizer(entry['question'])['input_ids']
                tokens = tokens[-self.max_length:]
                if len(tokens) < self.max_length:
                    tokens = tokens + [0] * (self.max_length - len(tokens))

            utils.assert_eq(len(tokens), self.max_length)
            entry['q_token'] = tokens
            

            entry['c_token'] = []
            choices = entry['choices']
            for choice in choices.values():
                c_tokens = self.dictionary.tokenize(choice, False)
                c_tokens = c_tokens[:self.max_choice_length]
                if len(c_tokens) < self.max_choice_length:
                    # pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (self.max_choice_length - len(c_tokens))
                    c_tokens = padding + c_tokens
                utils.assert_eq(len(c_tokens), self.max_choice_length)
                entry['c_token'].append(c_tokens)

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            if self.name in ['train', 'val']:
                assert isinstance(entry['answer_label'], int) # 0-3

            cTokens = torch.LongTensor(self.c_num, self.max_choice_length).zero_()

            choice_num = len(entry['c_token']) # number of current choices, e.g., 3
            c_token = torch.from_numpy(np.array(entry['c_token']))
            cTokens[0:choice_num, :].copy_(c_token)
            entry['c_token'] = cTokens

    def __getitem__(self, index):
        entry = self.entries[index]

        features = self.features[int(self.img2index[entry['image_name']])]

        question_id = entry['question_id']
        question = entry['q_token']
        choices = entry['c_token']
        img_name = entry['image_name']

        obj, desc_feat = self.info_embed[img_name] # [obj_max_num, 64], [obj_max_num, 1]


        if self.name in ['train', 'val']:
            answer = entry['answer']
            answer_label = entry['answer_label'] # 0-3
            assert ord(answer) - 97 == answer_label

            target = torch.zeros(self.c_num)
            if answer_label in range(self.c_num):
                target[answer_label] = 1.0
            
            return features, question, choices, target, obj, desc_feat, question_id
        elif self.name == "test":
            return features, question, choices, obj, desc_feat, question_id

    def __len__(self):
        return len(self.entries)