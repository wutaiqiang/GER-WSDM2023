
from genericpath import exists
import os
import codecs
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

class LoggerWithDepth():
    def __init__(self, log_dir, config=None,nickname=None) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        #sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        # path to write log
        self.sub_dir = os.path.join(self.log_dir, nickname)#sub_name+nickname)
        if os.path.exists(self.sub_dir):
            raise Exception("Logging Directory {} Has Already Exists. Change to another sub name or set OVERWRITE to True".format(self.sub_dir))
        else:
            os.makedirs(self.sub_dir, exist_ok=True)
        #self.checkpoint_path = os.path.join(self.sub_dir,"pytorch_model.bin")
        self.checkpoint_path = os.path.join(self.sub_dir,"pytorch_model.bin")

        # Setup File/Stream Writer
        log_format=logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s")
        self.writer = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.sub_dir, "training.log"))
        fileHandler.setFormatter(log_format)
        self.writer.addHandler(fileHandler)
        self.writer.setLevel(logging.INFO)
        # Setup tensorboard Writer
        self.painter = SummaryWriter(self.sub_dir)

        self.write_description_to_folder(os.path.join(self.sub_dir, 'params.txt'), config)
    
    def write_description_to_folder(self, file_name, config):
        with codecs.open(file_name, 'w') as desc_f:
            desc_f.write("- Training Parameters: \n")
            for key, value in config.__dict__.items():
                desc_f.write("  - {}: {}\n".format(key, value))


def set_random_seed(seed):
    """
    Set Random Seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
                yield sample_idx #! return idx
            
            if len(sample_idx) < self.batch_size:
                world_names.remove(world_name)
            world_value['pointer'] += self.batch_size
    
    def __len__(self):
        return sum([len(value) // self.batch_size + 1 for _, value in self.subworld_idx.items()])
    
class SubWorldDistributedSampler(DistributedSampler):
    def __init__(self, batch_size, subworld_idx, num_replicas, rank):
        self.batch_size = batch_size
        self.subworld_idx = subworld_idx
        self.num_replicas = num_replicas # total gpu num
        self.rank = rank # current gpu

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

