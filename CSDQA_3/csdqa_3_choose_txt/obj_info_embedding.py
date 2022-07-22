import os
import sys
import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer
from transformers import BertModel

sys.path.append("/home/chenyang/code/CCKS2022/IconQA")

from models.vitrm_models.PositionalEncoding import weight_embedding

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
        self.bertmodel = BertModel.from_pretrained(bert_models[lang_model]).cuda(0)
        # self.img2obj = json.load(open(os.path.join(dataroot, "img2obj.json")))
        self.diagram_detection = json.load(open(os.path.join(dataroot, "diagram_detection.json")))
        self.img2size = json.load(open(os.path.join(dataroot, 'img2size.json')))
        # self.obj_set = json.load(open(os.path.join(dataroot, 'obj_set.json')))
        # self.obj_label = self.obj_set["obj_label"]
        # self.shape_name = self.obj_set["shape_name"]
        # self.le_label = LabelEncoder().fit(self.obj_label)
        # self.le_name = LabelEncoder().fit(self.shape_name)
        self.obj_max_num = obj_max_num
        self.weight_region, self.offset = self.get_weight_region()
        self.weight_embedding = weight_embedding(self.weight_region + 1, 64)

    def __getitem__(self, img_name):
        region_list = self.diagram_detection[img_name]
        pic_width, pic_height = self.img2size[img_name]
        
        feat = torch.zeros(self.obj_max_num, 64)
        value_feat = torch.zeros(self.obj_max_num, 512)

        region_num = 0

        for region in region_list:
            if len(region) == 0:
                break
            if region_num >= self.obj_max_num:
                break
            shape_attributes = region["shape_attributes"]
            # name = shape_attributes['name']
            x = shape_attributes['x']
            y = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']
            region_attributes = region["region_attributes"]
            obj_id = region_attributes['Obj_id']
            # obj_label = region_attributes.get('Obj_label', '')
            # desc = region_attributes.get('Description', '')
            value = region_attributes['value']


            # name_feat = np.zeros((3))
            # name_feat[self.le_name.transform([name])[0]] = 1
            # name_feat = torch.from_numpy(name_feat)     # [3]

            # label_feat = np.zeros((57))
            # if obj_label == '':
            #     label_feat[56] = 1
            # else:
            #     label_feat[self.le_label.transform([obj_label])[0]] = 1
            # label_feat = torch.from_numpy(label_feat)   # [57]

            x_relative = x / pic_width
            y_relative = y / pic_height
            width_relative = width / pic_width
            height_relative = height / pic_height
            cord_feat = torch.tensor([x_relative, y_relative, width_relative, height_relative])     # [4]

            # id_feat = torch.tensor([int(obj_id)])       # [1]
            
            # if desc != '':
            #     tokenized_text = self.tokenizer.tokenize(desc)
            #     indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            #     segments_ids = [1] * len(tokenized_text)
            #     tokens_tensor = torch.tensor([indexed_tokens]).long().cuda(1)
            #     segments_tensors = torch.tensor([segments_ids]).long().cuda(1)
            #     desc_feat = self.bertmodel(tokens_tensor, segments_tensors)[1]     # [512]
            #     desc_feat = desc_feat.squeeze(0).cpu().detach()
            # else:
            #     desc_feat = torch.zeros((bert_dims[self.lang_model]))

            # if desc != '':
            #     index = desc.rfind("is") + 3
            #     assert index < desc.__len__(), print(info)
            #     if desc[index] == '-' or desc[index] in ['0','1','2','3','4','5','6','7','8','9']:
            #         if desc[index] == '-' and desc[index+1] not in ['0','1','2','3','4','5','6','7','8','9']:
            #             desc_weight = 0
            #         else:
            #             try:
            #                 desc_weight = int(desc[index:-1]) + self.offset + 1
            #             except ValueError:
            #                 desc_weight = 0
            #     else:
            #         desc_weight = 0
            # else:
            #     desc_weight = 0
            # desc_weight = torch.tensor([desc_weight])  # [1]
            # # desc_feat = self.weight_embedding(desc_weight)              # [64]

            value_ids = [self.tokenizer.convert_tokens_to_ids(value)]
            segments_ids = [1]
            tokens_tensor = torch.tensor([value_ids]).long().cuda(0)
            segments_tensors = torch.tensor([segments_ids]).long().cuda(0)
            value_weight = self.bertmodel(tokens_tensor, segments_tensors)[0]  # [1, 1, 512]
            value_weight = value_weight.squeeze()       # [512]



            # region_feat = torch.cat([name_feat, label_feat, cord_feat])  # [64]
            region_feat = cord_feat

            feat[region_num][:].copy_(region_feat)
            value_feat[region_num][:].copy_(value_weight)

            region_num += 1
        
        return feat, value_feat


    # def get_weight_region(self):
    #     obj_json = json.load(open("../data/CSDQA_3/img2obj.json"))
    #     num = 0
    #     count = 0
    #     min_num = 1e9
    #     max_num = 0
    #     for pic, pic_info in obj_json.items():
    #         for info in pic_info[:self.obj_max_num]:
    #             count += 1
    #             desc = info["region_attributes"]['Description']
    #             if desc != '':
    #                 index = desc.rfind("is") + 3
    #                 assert index < desc.__len__(), print(info)
    #                 if desc[index] == '-' or desc[index] in ['0','1','2','3','4','5','6','7','8','9']:
    #                     num += 1
    #                     if desc[index] == '-' and desc[index+1] not in ['0','1','2','3','4','5','6','7','8','9']:
    #                         pass
    #                     else:
    #                         try:
    #                             if max_num < int(desc[index:-1]):
    #                                 max_num = int(desc[index:-1])
    #                             if min_num > int(desc[index:-1]):
    #                                 min_num = int(desc[index:-1])
    #                         except ValueError:
    #                             pass
    #             else:
    #                 continue
    #     return max_num - min_num + 1, -min_num
        

if __name__ == '__main__':
    print(sys.path)
    
    info = Info_Embedding('bert-small', "../data/CSDQA_3", 55)
    print(info["11.png"])