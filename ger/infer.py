import argparse
from ast import arg 
import os 
from tqdm import tqdm

import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

import json

from transformers import BertTokenizerFast,AdamW, get_linear_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys

from ger.data_loader import ZeshelDataset, cross_collate_fn
from ger.utils import LoggerWithDepth, set_random_seed, SubworldBatchSampler, SubWorldDistributedSampler
from ger.model import BiEncoder, NCE_Random
from ger.evaluate import evaluate_bi_model

def argument_parser():
    """
    Initial params
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='data/zeshel')
    parser.add_argument('--pretrained_model', type=str, default="bert-base-uncased")
    parser.add_argument('--log_dir', type=str, default="logs1")
    parser.add_argument('--name', type=str, default="infer")
    parser.add_argument('--cache_dir', type=str, default="model_cash") 
    parser.add_argument('--entity_save_dir', type=str, default="logs/entity_cash") 
    parser.add_argument('--bi_ckpt_path', type=str, default="logs/hgat_dualloss_mu01/pytorch_model.bin") 

    parser.add_argument('--max_cand_len', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--handle_batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--eval_batch_size', type=int, default=12)
    parser.add_argument('--encode_batch_size', type=int, default=40, help="batchsize for encode the entity")
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float,  default=0.01)
    parser.add_argument('--warmup_ratio', type=float,  default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--eval_interval', type=int, default=100) 
    parser.add_argument('--seed', type=int, default=10086)
    parser.add_argument('--logging_interval', type=int, default=100)

    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--feat_drop', type=float, default=0.0) 
    parser.add_argument('--node_drop', type=float, default=0.0)
    parser.add_argument('--mu', type=float, default=0.1)

    parser.add_argument('--return_type', type=str, default="bert_only", help="types to return, bert_only, gnn .etc")
    parser.add_argument('--dual_loss', action="store_true")

    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_test', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument('--graph', action="store_true")

    return parser.parse_args()


def main(local_rank, args, train_dataset, valid_dataset, test_dataset, tokenizer):
    args.local_rank = local_rank
    is_master = local_rank in [-1,0] #master process
    if is_master:
        logger = LoggerWithDepth(args.log_dir,config=args,nickname=args.name)
    else:
        logger = None
        
    args.entity_save_dir = os.path.join(os.path.join(args.log_dir, args.name),"entity") # path to save entity
    os.makedirs(args.entity_save_dir, exist_ok=True)
    
    # Set Training Device
    if args.data_parallel:
        if args.n_gpu == 1:
            args.data_parallel = False
        else:    
            dist.init_process_group("nccl", rank=args.local_rank, world_size=args.n_gpu)
            torch.cuda.set_device(args.local_rank)
    
    args.device = "cuda" if not args.no_cuda else "cpu"
    set_random_seed(args.seed) #? +local_rank? 

    # Model Intial
    bi_model = BiEncoder(args).to(args.device)
    criterion = NCE_Random(args.n_gpu, dual_loss=args.dual_loss)
    
    # load checkpoint
    if args.bi_ckpt_path is not None:
        state_dict = torch.load(args.bi_ckpt_path, map_location='cpu')
        new_state_dict = {}
        for param_name, param_value in state_dict.items():
            if param_name[:7] == 'module.':
                new_state_dict[param_name[7:]] = param_value
            else:
                new_state_dict[param_name] = param_value
        bi_model.load_state_dict(new_state_dict)
        if is_master:
            logger.writer.info("Loading ckpt from {}".format(args.bi_ckpt_path))
    
    if args.n_gpu > 1:
        bi_model = DDP(bi_model, device_ids=[args.local_rank], find_unused_parameters=True) #DDP
    
    with torch.no_grad():
        max_score_test, best_candiate_test = evaluate_bi_model(bi_model, test_dataset, args, 
                save_dir=args.entity_save_dir, 
                mode="train", return_all_score=False, logger=logger)

    with open(os.path.join(logger.sub_dir, "train_candidate_@{:.2f}".format(max_score_test)), 'w') as f:
        for candidate in best_candiate_test:
            f.write(json.dumps(candidate) + '\n')

    del bi_model



if __name__ == "__main__":
    args = argument_parser()
    print(args.__dict__)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "18101"

    args.n_gpu = torch.cuda.device_count()

    #! before multiprocessing, preprocess the data
    train_dataset, valid_dataset, test_dataset = None, None, None
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model,cache_dir=args.cache_dir)

    

    
    test_dataset = ZeshelDataset(
            mode='train',
            args=args,
            tokenizer=tokenizer,
        )
    
    #! Main
    if args.n_gpu <= 1:
        main(0, args, train_dataset, valid_dataset, test_dataset, tokenizer,)
    else:
        mp.spawn(main, args=(args, train_dataset, valid_dataset, test_dataset, tokenizer,), nprocs=args.n_gpu, join=True)
    #main(0, args, train_dataset, valid_dataset, test_dataset, tokenizer,)