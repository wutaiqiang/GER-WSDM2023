
import argparse
import os
from select import select
from transformers import BertModel, BertConfig, AdamW, BertTokenizerFast, get_linear_schedule_with_warmup
import json
import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, Dataset
from tqdm import tqdm


import sys


from ger.data_loader import ZeshelDataset
from ger.utils import LoggerWithDepth, set_random_seed

def argument_parser():
    """
    Initial params
    """
    parser = argparse.ArgumentParser()
    #! read data
    parser.add_argument('--dataset_path', type=str, default='data/zeshel')
    parser.add_argument('--cache_dir', type=str, default="model_cash")
    parser.add_argument('--pretrained_model', type=str, default="bert-base-uncased")
    parser.add_argument('--max_cand_len', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--handle_batch_size', type=int, default=128)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument('--name', type=str, default="debug")
    parser.add_argument('--bi_ckpt_path', type=str, default=None)

    parser.add_argument('--train_path', type=str, default='logs1/bert_only_again/train_candidate_@47215.00')
    parser.add_argument('--valid_path', type=str, default='logs1/bert_only_again/valid_candidate_@9031.00')
    parser.add_argument('--test_path', type=str, default='logs1/bert_only_again/test_candidate_@8315.00')
    parser.add_argument('--log_dir', type=str, default="logs2")

    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--encode_batch_size', type=int, default=20, help="batchsize for encode the entity")
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float,  default=0.01)
    parser.add_argument('--warmup_ratio', type=float,  default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--eval_interval', type=int, default=100) 
    parser.add_argument('--seed', type=int, default=10086)
    parser.add_argument('--logging_interval', type=int, default=100)


    return parser.parse_args()

class CrossDataset(Dataset):
    def __init__(self, mode, args, tokenizer, candidate_path):
        ori_dataset = ZeshelDataset(
            mode=mode,
            args=args,
            tokenizer=tokenizer,
        )
        self.m = []
        self.e = []
        self.pos = []

        # build title:id
        title_to_token = {}
        for k,v in ori_dataset.entity_desc.items():
            d = {}
            for x in v:
                d[x["title"]] = x["token_ids"]
            title_to_token[k] = d

        #Read Candidate Set
        cands = []
        with open(candidate_path, "r") as f:
            for line in f.readlines():
                cands.append(json.loads(line))
        for sample, cand in tqdm(zip(ori_dataset.samples, cands), desc="read dataset"):
            world = sample["world"]
            label = sample["label"]
            gt = ori_dataset.entity_desc[world][label]
            cand_title = [x["title"] for x in cand][:args.top_k]
            if gt["title"] in cand_title:
                self.e.append([title_to_token[world].get(x, [0]*128)[1:-1]+[0, 0] for x in cand_title])
                self.m.append(sample["tokens"])
                self.pos.append(cand_title.index(gt["title"]))
        
        #! after pick
        self.ori_len = len(ori_dataset.samples)
        self.recall_len = len(self.m)
        print("{}: {}/{}".format(mode, self.recall_len, self.ori_len))
    
    def __getitem__(self, index):
        return {"m":self.m[index], "e":self.e[index], "pos": self.pos[index]}
    
    def __len__(self):
        return self.recall_len

def cross_collate_fn(batch):
    """
    aligned the sample from entity
    """
    ids_m = torch.tensor([sample['m'] for sample in batch])
    ids_e = torch.tensor([sample['e'] for sample in batch])

    pos = torch.tensor([sample['pos'] for sample in batch])

    return torch.cat((ids_m.unsqueeze(1).repeat(1,ids_e.size(1),1), ids_e), dim=-1), pos
    #return ids_m, ids_e, pos

class CrossEncoder(nn.Module):
    def __init__(self, args):
        super(CrossEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(args.pretrained_model, \
            return_dict = True, cache_dir=args.cache_dir) #output_attentions = True
        bert_output_dim = self.encoder.embeddings.word_embeddings.weight.size(1)
        self.linear1 = nn.Linear(bert_output_dim, 1)
    
    def to_bert_input(self, input_ids):
        attention_mask = 1 - (input_ids == 0).long()
        token_type_ids = torch.zeros_like(input_ids).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
    def forward(self, input, label):
        bsz, num, seq_len = input.size(0), input.size(1), input.size(2)
        bert_output = self.encoder(**self.to_bert_input(input.reshape(-1, seq_len)))
        vector_all = bert_output.last_hidden_state
        logits = self.linear1(vector_all[:,0,:]).unsqueeze(-1).reshape(bsz, num)
        loss = F.cross_entropy(logits, label, reduction="mean")
        predict = torch.max(logits, -1).indices
        acc = sum(predict == label) * 1.0 / bsz
        return logits, loss, acc

def evalute(model, dataset, args):
    is_master = args.local_rank in [-1,0]
    model.eval()
    # for ddp model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        test_module = model.module
    else:
        test_module = model
    
    if args.data_parallel:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    train_loader = DataLoader(dataset, sampler=sampler, collate_fn=cross_collate_fn, batch_size=args.eval_batch_size)
    hit, all = 0,0
    if is_master:
        iter_ = tqdm(train_loader)
    else:
        iter_ = train_loader
    with torch.no_grad():
        for batch in iter_:      
            ids, label = batch
            ids = ids.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # forward  
            logits, loss, acc = test_module(ids, label)
            all += logits.size(0)
            predict = torch.max(logits, -1).indices
            hit += sum(predict == label).item()
    
    path = os.path.join(args.log_dir, args.name)
    torch.save([all, hit], os.path.join(path, "{}.pt".format(args.local_rank)))

    if args.n_gpu > 1:
        torch.distributed.barrier()

    #! 非主进程直接退出
    if not is_master:
        return 0, 0
    
    A, H = 0,0
    for i in range(args.n_gpu):
        all, hit = torch.load(os.path.join(path, "{}.pt".format(i)))
        A += all
        H += hit
    return H, A

def main(local_rank, args, train_dataset, valid_dataset, test_dataset, tokenizer):
    args.local_rank = local_rank
    is_master = local_rank in [-1,0] # master process
    if is_master:
        logger = LoggerWithDepth(args.log_dir,config=args,nickname=args.name)
    else:
        logger = None
    
    # Set Training Device
    if args.data_parallel:
        if args.n_gpu == 1:
            args.data_parallel = False
        else:    
            dist.init_process_group("nccl", rank=args.local_rank, world_size=args.n_gpu)
            torch.cuda.set_device(args.local_rank)

    args.device = "cuda" if not args.no_cuda else "cpu"
    set_random_seed(args.seed) #? +local_rank? 

    #! Model Initial
    bi_model = CrossEncoder(args).to(args.device)

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

    # DataLoader
    if args.data_parallel:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=sampler, collate_fn=cross_collate_fn, batch_size=args.train_batch_size)
    

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {'params': [p for n, p in bi_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in bi_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    train_batch_size = args.train_batch_size // args.gradient_accumulation
    total_steps = len(train_dataset) * args.epoch // train_batch_size 
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if is_master:
        logger.writer.info("Optimization steps = {},  Warmup steps = {}".format(total_steps, warmup_steps))

    #hit, all = evalute(bi_model, valid_dataset, args)
    #if is_master:
    #    print("all {} hit {}".format(all, hit))

    #! Forward
    step = 0
    with tqdm(total = total_steps) as pbar:
        for e in range(args.epoch):
            tr_loss = []
            for batch in train_loader:
                bi_model.train()
                step += 1
                ids, label = batch
                ids = ids.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                # forward  
                logits, loss, acc = bi_model(ids, label)
                tr_loss.append(loss.item())
                loss.backward()
                        
                #if args.n_gpu > 1:
                #    dist.all_reduce(loss.div_(args.n_gpu))
                if step % args.gradient_accumulation == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(bi_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    bi_model.zero_grad()
                    
                if is_master: # record
                    #logger.writer.info("epoch: {}, loss: {:.4f}, acc: {:.4f}".format(e + 1, loss.item(), bi_acc.item()))
                    #! add to tensorboard
                    logger.painter.add_scalar('train_loss',loss.item(),step)
                    logger.painter.add_scalar('train_acc', acc.item(),step)
                        
                    if step % args.logging_interval == 0:
                        logger.writer.info("Step {}: Average Loss = {}".format(step, sum(tr_loss) / len(tr_loss)))
                        tr_loss = []
                
                if step % args.eval_interval == 0:
                    hit, all = evalute(bi_model, valid_dataset, args)
                    if is_master:
                        logger.painter.add_scalar('valid_acc', hit/all, step)
                        logger.painter.add_scalar('valid_all_acc', hit/valid_dataset.ori_len, step)
                        logger.writer.info("Step {}: valid_acc = {}".format(step, hit/all))
                        logger.writer.info("Step {}: valid_all_acc = {}".format(step, hit/valid_dataset.ori_len))
                    hit, all = evalute(bi_model, test_dataset, args)
                    if is_master:
                        logger.painter.add_scalar('test_acc', hit/all, step)
                        logger.painter.add_scalar('test_all_acc', hit/test_dataset.ori_len, step)
                        logger.writer.info("Step {}: test_acc = {}".format(step, hit/all))
                        logger.writer.info("Step {}: test_all_acc = {}".format(step, hit/test_dataset.ori_len))
    
    hit, all = evalute(bi_model, valid_dataset, args)
    if is_master:
        logger.painter.add_scalar('valid_acc', hit/all, step)
        logger.painter.add_scalar('valid_all_acc', hit/valid_dataset.ori_len, step)
        logger.writer.info("Step {}: valid_acc = {}".format(step, hit/all))
        logger.writer.info("Step {}: valid_all_acc = {}".format(step, hit/valid_dataset.ori_len))
    hit, all = evalute(bi_model, test_dataset, args)
    if is_master:
        logger.painter.add_scalar('test_acc', hit/all, step)
        logger.painter.add_scalar('test_all_acc', hit/test_dataset.ori_len, step)
        logger.writer.info("Step {}: test_acc = {}".format(step, hit/all))
        logger.writer.info("Step {}: test_all_acc = {}".format(step, hit/test_dataset.ori_len))
    
    del optimizer

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

    train_dataset = CrossDataset("train", args, tokenizer, args.train_path)
    valid_dataset = CrossDataset("valid", args, tokenizer, args.valid_path)
    test_dataset = CrossDataset("test", args, tokenizer, args.test_path)
    
    #! Main
    if args.n_gpu <= 1:
        main(0, args, train_dataset, valid_dataset, test_dataset, tokenizer,)
    else:
        mp.spawn(main, args=(args, train_dataset, valid_dataset, test_dataset, tokenizer,), nprocs=args.n_gpu, join=True)
