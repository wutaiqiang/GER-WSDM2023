#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from ast import arg
import os
import json
import random
from tqdm import tqdm
import numpy as np 
import nltk
from time import time 
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from allennlp.predictors.predictor import Predictor, load_archive

import multiprocessing
import torch.multiprocessing as mp
from allennlp.common.util import lazy_groups_of

import sys
sys.path.append('GER-WSDM2023) #! parent fold here
from ger.extract import extract_kg, extract_kg_batch

WORLDS = {
    'train': [("american_football", 31929), ("doctor_who", 40281), ("fallout", 16992), ("final_fantasy", 14044), ("military", 104520), ("pro_wrestling", 10133), ("starwars", 87056), ("world_of_warcraft", 27677)],
    'valid': [("coronation_street", 17809), ("muppets", 21344), ("ice_hockey", 28684), ("elder_scrolls", 21712)],
    'test': [("forgotten_realms", 15603), ("lego", 10076), ("star_trek", 34430), ("yugioh", 10031)]
}


CLS_TAG = "[CLS]"
SEP_TAG = "[SEP]"
MENTION_START_TAG = "[unused0]"
MENTION_END_TAG = "[unused1]"
ENTITY_TAG = "[unused2]"

class SubworldBatchSampler(Sampler):
    def __init__(self, batch_size, subworld_idx):
        self.batch_size = batch_size
        self.subworld_idx = subworld_idx

    def __iter__(self):
        for world_name, world_value in self.subworld_idx.items():
            world_value['perm_idx'] = torch.randperm(len(world_value['idx'])) #random generate idx
            world_value['pointer'] = 0
        world_names = list(self.subworld_idx.keys())

        while len(world_names) > 0:
            world_name = np.random.choice(world_names) #random choice
            world_value = self.subworld_idx[world_name]
            start_pointer = world_value['pointer'] # start from 0
            sample_perm_idx = world_value['perm_idx'][start_pointer:start_pointer + self.batch_size]
            sample_idx = [world_value['idx'][idx] for idx in sample_perm_idx] #the index in whole dataset

            if len(sample_idx) > 0:
                yield sample_idx
            
            if len(sample_idx) < self.batch_size:
                world_names.remove(world_name)
            world_value['pointer'] += self.batch_size
    
    def __len__(self):
        return sum([len(value) // self.batch_size + 1 for _, value in self.subworld_idx.items()])
    
class SubWorldDistributedSampler(DistributedSampler):
    def __init__(self, batch_size, subworld_idx, num_replicas, rank):
        self.batch_size = batch_size
        self.subworld_idx = subworld_idx
        self.num_replicas = num_replicas
        self.rank = rank

        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        for world_name, world_value in self.subworld_idx.items():
            world_value['perm_idx'] = torch.randperm(len(world_value['idx']), generator=g).tolist()
            world_value['pointer'] = 0
        world_names = list(self.subworld_idx.keys())

        while len(world_names) > 0:
            world_idx = torch.randint(len(world_names), size=(1, ), generator=g).tolist()[0]
            world_name = world_names[world_idx]
            
            world_value = self.subworld_idx[world_name]
            start_pointer = world_value['pointer']
            sample_perm_idx = world_value['perm_idx'][start_pointer : start_pointer + self.batch_size]

            if len(sample_perm_idx) == 0:
                world_names.remove(world_name)
                continue
            
            if len(sample_perm_idx) < self.batch_size :
                world_names.remove(world_name)
                sample_perm_idx = sample_perm_idx + world_value['perm_idx'][:self.batch_size - len(sample_perm_idx)]
            #print(self.rank, sample_perm_idx)
            sample_perm_idx = sample_perm_idx[self.rank::self.num_replicas] # divide data into multi device
            
            try:
                sample_idx = [world_value['idx'][idx] for idx in sample_perm_idx]
                assert len(sample_idx) == self.batch_size // self.num_replicas
            except:
                print(world_name, sample_perm_idx, sample_idx, len(world_value['idx']))
            yield sample_idx
            world_value['pointer'] += self.batch_size
        
        self.epoch += 1

    #def __len__(self):
    #    return sum([len(value) // self.batch_size + 1 for _, value in self.subworld_idx.items()])

class EncodeDataset(Dataset):
    """
    dataset for description text
    """
    def __init__(self, document_path, world, tokenizer, max_seq_len, predictor, debug=False, cand_batch_size=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.world = world
        self.predictor = predictor
        self.debug = debug
        self.cand_batch_size = cand_batch_size
        # cash for data

        preprocess_path = os.path.join(document_path, 'preprocess1')

        file_name = os.path.join(preprocess_path, world + '.jsonl')

        if os.path.exists(preprocess_path) and os.path.exists(file_name):
            #self.samples = torch.load(file_name)
            with open(file_name, 'r') as f:
                self.samples = [json.loads(line) for line in f]
        else:
            if not os.path.exists(preprocess_path):
                os.mkdir(preprocess_path)
            
            document_path = os.path.join(document_path, world + '.json')
            # load entity description text
            self.samples = self.load_entity_description(document_path, world)
            #torch.save(self.samples, file_name)
            with open(file_name, 'w') as f:
                for sample in self.samples:
                    f.write(json.dumps(sample) + '\n')

        
    def __len__(self):
        return len(self.samples)
    
    def get_nth_title(self, idx):
        return self.samples[idx]['title']
    
    def load_entity_description(self, document_path, world):
        """
        load entity description text
        """
        tokenizer = self.tokenizer
        max_seq_len = self.max_seq_len

        res = []

        all_title_text = []
        all_title = []
        all_title_node_mask = []
        all_desp = []

        num_lines = sum(1 for line in open(document_path, 'r'))
        print("World/{}: preprocessing {} samples".format(world, num_lines))

        with open(document_path, 'r') as f:
            for idx, line in enumerate(tqdm(f, total=num_lines)):
                if self.debug and idx > 200:
                    break
                info = json.loads(line)
                title = info['title'].lower()
                all_title_text.append(title)
                description = info['text'].lower() # cut off text here to save memory
                title_id = tokenizer.convert_tokens_to_ids([CLS_TAG] + tokenizer.tokenize(title) + [ENTITY_TAG])
                all_title_node_mask.append([1,len(title_id)-2]) # view title as node 0
                des_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(description))
                des_id = des_id[:max_seq_len - len(title_id)]
                all_title.append(title_id)
                all_desp.append(des_id)

        idx = 0
        for batch_desp in tqdm(lazy_groups_of(all_desp, self.cand_batch_size),total=(len(all_desp)//self.cand_batch_size+1)):
            batch_des_ids, batch_node_mask, batch_rels = extract_kg_batch(batch_desp, tokenizer, self.predictor, print_info=False)
            for i,des_ids in enumerate(batch_des_ids):
                title_id = all_title[idx]
                ids = title_id+des_ids
                #! shift
                node_mask = [all_title_node_mask[idx]]+[[pos[0]+len(title_id),pos[1]+len(title_id)]for pos in batch_node_mask[i]]
                rels = batch_rels[i]

                if len(ids) < max_seq_len:
                    ids += [0] *(max_seq_len - len(ids))
                elif len(ids) > max_seq_len:
                    ids = ids[:max_seq_len]
                    # the leake node
                    over_node = [idxx for idxx,m in enumerate(node_mask) if m[0] >= max_seq_len]
                    rels = [rel for rel in rels if over_node not in rel]
                
                #ids += tokenizer.convert_tokens_to_ids([SEP_TAG])
                #assert max([max(pos) for pos in node_mask]) < max_seq_len, print("pos wrong!", title)
                #assert max([max(pos) for pos in rels])+1==len(node_mask), print("rels wrong", rels, node_mask)
                res.append(
                    {
                        "token_ids": ids,
                        "title": all_title_text[idx],
                        "node_mask": node_mask,
                        "rels": rels,
                    }
                )
                idx += 1
        return res
    
    def __getitem__(self, idx):
        return self.samples[idx]


class ZeshelDataset(Dataset):
    def __init__(self, mode, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        #! Allen NLP
        self.predictor = Predictor.from_path(os.path.join(args.cache_dir,"openie-model.2020.03.26.tar.gz"), cuda_device=0)

        self.entity_desc = {
            world[0]: EncodeDataset(
                document_path = os.path.join(args.dataset_path, 'documents'),
                world = world[0],
                tokenizer = self.tokenizer,
                max_seq_len = self.args.max_cand_len,
                predictor=self.predictor,
                debug=self.args.debug,
                cand_batch_size=self.args.handle_batch_size
            ) #!initial data for each world description
            for world in WORLDS[mode]
        }

        self.load_training_samples(os.path.join(args.dataset_path, 'blink_format'), mode)

        #! few sample
        if self.args.train_ratio != 1.0 and "train" in self.mode:
            random.seed(int(self.args.train_ratio*100))
            self.samples = random.sample(self.samples, int(self.args.train_ratio*len(self.samples)))
            print("Using {} training data".format(self.args.train_ratio))

        self.subworld_idx = self.get_subworld_idx()

    def get_subworld_idx(self):
        # which idx in traning set is in which world
        worlds_sample_idx = {world[0]: {'idx': [], 'num': 0} for world in WORLDS[self.mode]}
        for idx, sample in enumerate(self.samples):
            world = sample['world']
            worlds_sample_idx[world]['idx'].append(idx) 
            worlds_sample_idx[world]['num'] += 1
        
        return worlds_sample_idx

    def load_training_samples(self, dataset_path, mode):
        """
        load the dataset for Mode
        """
        token_path = os.path.join(dataset_path, "{}_token1.jsonl".format(mode))
        # load or generate
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                self.samples = [json.loads(line) for line in f]
                print("Set/{}: Load {} samples".format(mode, len(self.samples)))
        else:
            data_path = os.path.join(dataset_path, "{}.jsonl".format(mode))
            num_lines = sum(1 for line in open(data_path, 'r'))

            raw_samples = []
            print("Set/{}: preprocessing {} samples".format(mode, num_lines))
            
            with open(data_path, 'r') as sample_f:
                for sample_line in tqdm(sample_f, total = num_lines):
                    raw_samples.append(json.loads(sample_line))
                    
            self.samples = self.handle_training_samples(raw_samples)

            with open(token_path, 'w') as f:
                for sample in self.samples:
                    f.write(json.dumps(sample) + '\n')

    def handle_training_samples(self, samples):
        if self.args.debug:
            samples = samples[:200]
        max_seq_len = self.args.max_seq_len
        # handle context
        all_input_ids = []
        all_label = []
        all_world = []
        mention_all = []
        guess_pos = []

        for sample in tqdm(samples):
            mention_tokens = []
            if sample['mention'] and len(sample['mention']) > 0:
                mention_tokens = self.tokenizer.tokenize(sample['mention'])
                #mention_tokens = [MENTION_START_TAG] + mention_tokens + [MENTION_END_TAG]
            mention_all.append(self.tokenizer.convert_tokens_to_ids(mention_tokens))


            context_left = sample["context_left"]
            context_right = sample["context_right"]
            context_left = self.tokenizer.tokenize(context_left)
            context_right = self.tokenizer.tokenize(context_right)

            left_quota = (max_seq_len - len(mention_tokens)- 2) // 2 - 1
            right_quota = max_seq_len - len(mention_tokens) - left_quota - 4
            left_add = len(context_left)
            right_add = len(context_right)
            if left_add <= left_quota:
                if right_add > right_quota:
                    right_quota += left_quota - left_add
            else:
                if right_add <= right_quota:
                    left_quota += right_quota - right_add

            guess_pos.append(left_quota)

            context_tokens = (
                context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
            )

            #context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
            #input_ids += [0]*(max_seq_len - len(input_ids))
            #assert len(input_ids) == max_seq_len

            all_input_ids.append(input_ids)
            all_label.append(sample['label_id'])
            all_world.append(sample['world'])
        
        # batch extract graph
        idx = 0 # idx for gloabl
        res = []
        for batch_sample in tqdm(lazy_groups_of(all_input_ids, self.args.handle_batch_size), total=len(all_input_ids)//self.args.handle_batch_size+1):
            
            batch_ids, batch_node_mask, batch_rels =extract_kg_batch(batch_sample,self.tokenizer,self.predictor, print_info=False)
            
            for i, ids in enumerate(batch_ids):
                node_mask = batch_node_mask[i]
                rels = batch_rels[i]
                
                #! handle mention node
                # find mention using:  ids,mention_all[idx],guess_pos[idx]
                pos = find_pos(ids,mention_all[idx],guess_pos[idx])
                if pos == -1: # Not find
                    # [CLS] as node 0
                    ids = self.tokenizer.convert_tokens_to_ids([CLS_TAG]) + ids
                    node_mask_new = [[0,0]]+[[pair[0]+1,pair[1]+1] for pair in node_mask]
                else:
                    # mention as node 0
                    ids = self.tokenizer.convert_tokens_to_ids([CLS_TAG]) + ids[:pos]+ \
                        self.tokenizer.convert_tokens_to_ids([MENTION_START_TAG]) + ids[pos:pos+len(mention_all[idx])] + \
                        self.tokenizer.convert_tokens_to_ids([MENTION_END_TAG]) + ids[pos+len(mention_all[idx]):]
                    node_mask_new = [[pos+2,pos+1+len(mention_all[idx])]]
                    for pair in node_mask:
                        new_pair = []
                        for x in pair:
                            if x < pos:
                                x += 1
                            elif x < pos+len(mention_all[idx]):
                                x += 2
                            else:
                                x += 3
                            new_pair.append(x)
                        node_mask_new.append(new_pair)
                
                # pad or cut for: ids,rels,node_mask_new
                if len(ids) < max_seq_len:
                    ids += [0] *(max_seq_len - len(ids))
                elif len(ids) > max_seq_len:
                    ids = ids[:max_seq_len]
                    # the leake node
                    over_node = [idxx for idxx,m in enumerate(node_mask_new) if m[0] >= max_seq_len]
                    rels = [rel for rel in rels if over_node not in rel]


                res.append(
                    {
                        "tokens": ids,
                        "node_mask": node_mask_new,
                        "rels": rels,
                        "label": all_label[idx],
                        "world": all_world[idx],
                    }
                )
                idx += 1 
            
        return res


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens_m = sample['tokens']
        node_mask_m = sample["node_mask"]
        rels_m = sample["rels"]
        label, world = sample['label'], sample['world']
        sample_e = self.entity_desc[world][label]
        tokens_e = sample_e["token_ids"]
        node_mask_e = sample_e["node_mask"]
        rels_e = sample_e["rels"]
        
        return {
            "ids_m": tokens_m,
            "ids_e": tokens_e,
            "node_mask_m": node_mask_m,
            "node_mask_e": node_mask_e,
            "rels_m": rels_m,
            "rels_e": rels_e,
            "world": world,
            "label_world_idx": label
        }


def find_pos(ids,mention,guess_pos):
    l = len(mention)
    for offset in range(max(guess_pos,len(ids)-guess_pos)):
        for delta in [-offset, offset]:
            new_pos = guess_pos + delta
            if new_pos >= 0 and new_pos < len(ids):
                if ids[new_pos:new_pos+l] == mention:
                    return new_pos
    return -1

def get_len(x):
    if not x: return 0
    return len(x)

def batch_pad(input,num=2):
    '''
    [A,B,C], pad the A B C
    '''
    length = max([get_len(x) for x in input]) # for rels, may []
    return [x+[[-1]*num]*(length-get_len(x)) for x in input]

def entity_collate_fn(batch):
    """
    aligned the sample from entity
    """
    title = [sample['title'] for sample in batch]
    ids_e = torch.tensor([sample['token_ids'] for sample in batch])
    node_mask_e = torch.tensor(batch_pad([sample['node_mask'] for sample in batch], num=2))
    rels_e = torch.tensor(batch_pad([sample['rels'] for sample in batch], num=3))

    return {
        'title': title,
        'entity':{"ids": ids_e, "node_mask":node_mask_e, "rels":rels_e}
    }


def cross_collate_fn(batch):
    """
    aligned the samples
    """
    world = [sample['world'] for sample in batch]
    label_world_idx = torch.tensor([sample['label_world_idx'] for sample in batch])
    
    ids_m = torch.tensor([sample['ids_m'] for sample in batch])
    ids_e = torch.tensor([sample['ids_e'] for sample in batch])
    node_mask_m = torch.tensor(batch_pad([sample['node_mask_m'] for sample in batch], num=2))
    node_mask_e = torch.tensor(batch_pad([sample['node_mask_e'] for sample in batch], num=2))
    rels_m = torch.tensor(batch_pad([sample['rels_m'] for sample in batch], num=3))
    rels_e = torch.tensor(batch_pad([sample['rels_e'] for sample in batch], num=3))


    return {
        'world': world,
        'label_world_idx': label_world_idx,
        'mention': {"ids": ids_m, "node_mask":node_mask_m, "rels":rels_m},
        'entity':  {"ids": ids_e, "node_mask":node_mask_e, "rels":rels_e}
    }


    