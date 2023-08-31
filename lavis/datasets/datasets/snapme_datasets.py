import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset

import json

from PIL import Image
import random
import torch
from lavis.datasets.data_utils import Vocabulary
import pickle

class SNAPMeDataset(BaseDataset): ## for VQA
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split):
        """
        vis_processor (string): visual processor
        text_processor (string): textual processor
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (string): Root directory of images (e.g. coco/images/)
        """
        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_root = vis_root
        self.split = split

        self.dataset = json.load(open(ann_paths[0], "r"))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        ingrs_vocab_v = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
        self.ingr2idx = ingrs_vocab_v.word2idx

        self.snapme_mapping = json.load(open("/nfs_share2/code/donghee/LAVIS/snapme_mapping.json", 'r'))
        wrong_nums = json.load(open("/nfs_share2/code/donghee/LAVIS/ingr_wrong.json", 'r'))
        self.wrong_nums = wrong_nums.keys()

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):

        sample = self.dataset[index]
        gt = sample['gt']
        gt_int = sample['gt_int']

        ## TODO gt -> gt int
        ## assert len(gt) == len(gt_int)

        random.shuffle(gt)
        answer = ', '.join(gt)
        answer = self.text_processor(answer)

        img_id = sample['id']
        img_path = os.path.join(self.vis_root, img_id)
        img = Image.open(img_path).convert('RGB')
        img = self.vis_processor(img)

        question = 'Question: What are the ingredients I need to make this food? Answer:'

        return {
            'image': img,
            'text_input': question,
            'text_output': answer,
            'ingredient': gt,
            'ingredient_int': gt_int,
            'img_id': img_id
            }
    
    def collater(self, data):

        img_list, input_list, output_list, ingr_list, ingr_int_list, img_id_list= [], [], [], [], [], []

        for sample in data:
            gt_int_set = set(sample['ingredient_int'])
            if len(gt_int_set.intersection(self.wrong_nums)) != 0: ## wrong num 포함하면, skip
                continue
            if sample is not None and sample['image'] is not None:
                img_list.append(sample['image'])
                input_list.append(sample['text_input'])
                output_list.append(sample['text_output'])
                ingr_list.append(sample['ingredient'])
                ingr_int_list.append(sample['ingredient_int'])
                img_id_list.append(sample['img_id'])
        
        return {
            'image' : torch.stack(img_list, dim=0),
            'text_input': input_list,
            'text_output': output_list,
            'ingredient': ingr_list,
            'ingredient_int': ingr_int_list,
            'img_id': img_id_list
            }


