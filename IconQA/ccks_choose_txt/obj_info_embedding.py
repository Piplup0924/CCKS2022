import os
import sys
import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer
from transformers import BertModel

bert_models = {'bert-base-uncased': 'bert-base-uncased',
               'bert-tiny':   'google/bert_uncased_L-2_H-128_A-2',
               'bert-mini':   'google/bert_uncased_L-4_H-256_A-4',
               'bert-small':  'google/bert_uncased_L-4_H-512_A-8',
               'bert-medium': 'google/bert_uncased_L-8_H-512_A-8',
               'bert-base':   'google/bert_uncased_L-12_H-768_A-12'}

bert_dims = {'bert-tiny':128, 'bert-mini':256, 'bert-small':512, 'bert-medium':512, 'bert-base':768, 'bert-base-uncased':768}

class Info_Embedding(object):
    def __init__(self, lang_model, dataroot, obj_max_num) -> None:
        self.lang_model = lang_model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_models[lang_model])
        self.bertmodel = BertModel.from_pretrained(bert_models[lang_model]).cuda(1)
        self.img2obj = json.load(open(os.path.join(dataroot, "img2obj.json")))
        self.img2size = json.load(open(os.path.join(dataroot, 'img2size.json')))
        self.obj_set = json.load(open(os.path.join(dataroot, 'obj_set.json')))
        self.obj_label = self.obj_set["obj_label"]
        self.shape_name = self.obj_set["shape_name"]
        self.le_label = LabelEncoder().fit(self.obj_label)
        self.le_name = LabelEncoder().fit(self.shape_name)
        self.obj_max_num = obj_max_num

    def __getitem__(self, img_name):
        region_list = self.img2obj[img_name]
        pic_width, pic_height = self.img2size[img_name]
        
        feat = torch.zeros(self.obj_max_num, 577)

        region_num = 0

        for region in region_list:
            if region_num >= self.obj_max_num:
                break
            shape_attributes = region["shape_attributes"]
            name = shape_attributes['name']
            x = shape_attributes['x']
            y = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']
            region_attributes = region["region_attributes"]
            obj_id = region_attributes['Obj_id']
            obj_label = region_attributes.get('Obj_label', '')
            desc = region_attributes.get('Description', '')

            name_feat = np.zeros((3))
            name_feat[self.le_name.transform([name])[0]] = 1
            name_feat = torch.from_numpy(name_feat)     # [3]

            label_feat = np.zeros((57))
            if obj_label == '':
                label_feat[56] = 1
            else:
                label_feat[self.le_label.transform([obj_label])[0]] = 1
            label_feat = torch.from_numpy(label_feat)   # [57]

            x_relative = x / pic_width
            y_relative = y / pic_height
            width_relative = width / pic_width
            height_relative = height / pic_height
            cord_feat = torch.tensor([x_relative, y_relative, width_relative, height_relative])     # [4]

            id_feat = torch.tensor([int(obj_id)])       # [1]
            
            if desc != '':
                tokenized_text = self.tokenizer.tokenize(desc)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids = [1] * len(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens]).long().cuda(1)
                segments_tensors = torch.tensor([segments_ids]).long().cuda(1)
                desc_feat = self.bertmodel(tokens_tensor, segments_tensors)[1]     # [512]
                desc_feat = desc_feat.squeeze(0).cpu().detach()
            else:
                desc_feat = torch.zeros((bert_dims[self.lang_model]))

            region_feat = torch.cat([name_feat, label_feat, cord_feat, id_feat, desc_feat])

            feat[region_num][:].copy_(region_feat)

            region_num += 1
        
        return feat



if __name__ == '__main__':
    info = Info_Embedding('bert-small', "../data/ccksdata", 55)
    print(info["11.png"])