from lavis.datasets.datasets.base_dataset import BaseDataset

import json
import pickle
import lmdb
import os
import numpy as np
import random
import torch
from PIL import Image
from ..data_utils import Vocabulary


class Recipe1MpDataset(BaseDataset):
    def __init__ (self, vis_processor, text_processor, vis_root, ann_paths, split, max_num_labels, max_num_samples=-1):
        
        self.split = split
        self.image_file = lmdb.open(os.path.join('/nfs_share2/code/donghee/inversecooking/data', 'lmdb_'+ split + '_1Mp'), max_readers=1, readonly=True, lock = False, readahead=False, meminit=False)

        # self.annotation = []
        # for ann_path in ann_paths:
        #     self.annotaion.extend(pickle.load(open(ann_path), 'rb')) ## TODO
        self.dataset = pickle.load(open(ann_paths[0], 'rb'))
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # self._add_instance_recipe_ids()

        self.ids = [] ## reciperecipe_id!!
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        self.ingrs_vocab = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
        self.instrs_vocab = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_toks.pkl', 'rb'))

        self.label2word = self.get_ingrs_vocab()

        self.maxnumims = 5
        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples]
        
        self.root = os.path.join('/home/donghee/im2recipe-Pytorch/data/images', self.split)
        self.max_num_labels = max_num_labels
    
    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)
    
    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient
    
    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[self.ids[index]]
        recipe_id = sample['id']
        # captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims]
        if self.split == 'train' or self.split == 'train':
            idx = np.random.randint(0, len(paths))
        else:
            idx = 0
        path = paths[idx]

        try:
            with self.image_file.begin(write=False) as txn:
                image = txn.get(path.encode())
                image = np.fromstring(image, dtype=np.uint8)
                image = np.reshape(image, (256, 256, 3))
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            image = self.vis_processor(image) ## TODO
        except:
            # print ("Image recipe_id not found in lmdb. Loading jpeg file...")
            # image = Image.open(os.path.join(self.root, path[0], path[1],
            #                                 path[2], path[3], path)).convert('RGB')
            image = None

        idx = index

        ingr_gt = self.dataset[self.ids[idx]]['ingredients']
        ingr_gt = ", ".join(ingr_gt)
        ingr_gt = ingr_gt.replace('_',' ')
        ingr_gt = self.text_processor(ingr_gt) ## TODO

        # labels = self.dataset[self.recipe_ids[recipe_idx]]['ingredients']
        title = sample['title']
        title = " ".join(title)

        # return image, ingr_gt, title, recipe_id, path
        return {
            'image': image,
            'text_input': ingr_gt,
            'title': title,
            'recipe_id': recipe_id,
            'img_id': path
        }


        # tokens = []
        # tokens.extend(title)
        # # add fake token to separate title from recipe
        # tokens.append('<eoi>')
        # for c in captions:
        #     tokens.extend(c)
        #     tokens.append('<eoi>')

        # ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        # pos = 0

        # true_ingr_recipe_idxs = []
        # for i in range(len(labels)):
        #     true_ingr_recipe_idxs.append(self.ingrs_vocab(labels[i]))

        # for i in range(self.max_num_labels):
        #     if i >= len(labels):
        #         label = '<pad>'
        #     else:
        #         label = labels[i]
        #     label_recipe_idx = self.ingrs_vocab(label)
        #     if label_recipe_idx not in ilabels_gt:
        #         ilabels_gt[pos] = label_recipe_idx
        #         pos += 1

        # ilabels_gt[pos] = self.ingrs_vocab('<end>')
        # ingrs_gt = torch.from_numpy(ilabels_gt).long()

        # if len(paths) == 0:
        #     path = None
        #     image_input = torch.zeros((3, 224, 224))
        # else:
        #     if self.split == 'train' or self.split == 'train':
        #         recipe_idx = np.random.randint(0, len(paths))
        #     else:
        #         recipe_idx = 0
        #     path = paths[recipe_idx]
        #     if self.use_lmdb:
        #         try:
        #             with self.image_file.begin(write=False) as txn:
        #                 image = txn.get(path.encode())
        #                 image = np.fromstring(image, dtype=np.uint8)
        #                 image = np.reshape(image, (256, 256, 3))
        #             image = Image.fromarray(image.astype('uint8'), 'RGB')
        #         except:
        #             # print ("Image recipe_id not found in lmdb. Loading jpeg file...")
        #             # image = Image.open(os.path.join(self.root, path[0], path[1],
        #             #                                 path[2], path[3], path)).convert('RGB')
        #             image_input = None
        #             caption = []

        #             caption = self.caption_to_recipe_idxs(tokens, caption)
        #             caption.append(self.instrs_vocab('<end>'))

        #             caption = caption[0:self.maxseqlen]
        #             target = torch.Tensor(caption)

        #             return image_input, target, ingrs_gt, recipe_id, path, self.instrs_vocab('<pad>')
        #             ## 여기꼭 원래대로!! root도 다시 돌려놓고

        #     else:
        #         image = Image.open(os.path.join(self.root, path[0], path[1], path[2], path[3], path)).convert('RGB')
        #     if self.transform is not None:
        #         image = self.transform(image)
        #     image_input = image

        # # Convert caption (string) to word recipe_ids. ## instructions
        # caption = []

        # caption = self.caption_to_recipe_idxs(tokens, caption)
        # caption.append(self.instrs_vocab('<end>'))

        # caption = caption[0:self.maxseqlen]
        # target = torch.Tensor(caption)

        # return image_input, target, ingrs_gt, recipe_id, path, self.instrs_vocab('<pad>')


    def collater(self, data):
        image_list, ingr_list, title_list, recipe_id_list, path_list = [], [], [], [], []

        for sample in data:
            if sample['image'] is not None:
                image_list.append(sample['image'])
                ingr_list.append(sample['text_input'])
                title_list.append(sample['title'])
                recipe_id_list.append(sample['recipe_id'])
                path_list.append(sample['img_id'])
        
        return {
            'image' : torch.stack(image_list, dim=0),
            'text_input': ingr_list,
            'title': title_list,
            'recipe_id': recipe_id_list,
            'img_id': path_list
        }


        # data = [sample for sample in data if sample['image'] is not None] ##
        
        # image, ingr_gt, title, recipe_id, path = zip(*data)

        # image = torch.stack(image, 0)
        # ingr_gt = torch.stack(ingr_gt, 0)

        # return image, ingr_gt, title, recipe_id, path
    
    # def collater(self, samples):
    #     image_list, question_list, answer_list, weight_list = [], [], [], []

    #     num_answers = []

    #     for sample in samples:
    #         image_list.append(sample["image"])
    #         question_list.append(sample["text_input"])

    #         weight_list.extend(sample["weights"])

    #         answers = sample["answers"]

    #         answer_list.extend(answers)
    #         num_answers.append(len(answers))

    #     return {
    #         "image": torch.stack(image_list, dim=0),
    #         "text_input": question_list,
    #         "answer": answer_list,
    #         "weight": torch.Tensor(weight_list),
    #         "n_answers": torch.LongTensor(num_answers),
    #     }
    


        