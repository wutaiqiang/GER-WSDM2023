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
# sys.path.append('GER-WSDM2023')

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
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--cache_dir', type=str, default="model_cache") 
    parser.add_argument('--entity_save_dir', type=str, default="logs/entity_cash") 
    parser.add_argument('--bi_ckpt_path', type=str, default=None) 

    parser.add_argument('--max_cand_len', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--handle_batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--eval_batch_size', type=int, default=12)
    parser.add_argument('--encode_batch_size', type=int, default=20, help="batchsize for encode the entity")
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
    parser.add_argument('--mu', type=float, default=0.5)

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

    num_params = sum(param.numel() for param in bi_model.parameters())
    print("Total Params: {} M".format(num_params/1e6))
    
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
    
    max_score, max_score_train, max_score_test = 0, 0, 0 #! record max score
    

    if args.do_train:
        train_batch_size = args.train_batch_size // args.gradient_accumulation
            
        if args.data_parallel:
            sampler = SubWorldDistributedSampler(batch_size=args.train_batch_size, subworld_idx=train_dataset.subworld_idx, num_replicas=args.n_gpu, rank=args.local_rank)
        else:
            sampler = SubworldBatchSampler(batch_size=args.train_batch_size, subworld_idx=train_dataset.subworld_idx)
        train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, collate_fn = cross_collate_fn)

        # optimizer & scheduler
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {'params': [p for n, p in bi_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in bi_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        total_steps = len(train_dataset) * args.epoch // train_batch_size 
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        if is_master:
            logger.writer.info("Optimization steps = {},  Warmup steps = {}".format(total_steps, warmup_steps))

        #! Forward
        step = 0
        with tqdm(total = total_steps) as pbar:
            for e in range(args.epoch):
                tr_loss = []
                for batch in train_dataloader:
                    bi_model.train()
                    step += 1

                    world = batch['world']
                    for w in world[1:]: 
                        assert world[0] == w
                    
                    mention, entity = batch["mention"], batch["entity"]
                    mention = {k:v.cuda(non_blocking=True) for k,v in mention.items()}
                    entity = {k:v.cuda(non_blocking=True) for k,v in entity.items()}
                    
                    #ctx_ids, ent_ids = batch['context_ids'], batch['label_ids']
                    #ctx_ids = ctx_ids.cuda(non_blocking=True) #speed up
                    #ent_ids = ent_ids.cuda(non_blocking=True)
                    ctx_output, ent_output = bi_model(mention, entity)
                    loss, bi_acc, bi_score = criterion(ctx_output, ent_output)
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
                        logger.painter.add_scalar('train_acc', bi_acc.item(),step)
                        
                        if step % args.logging_interval == 0:
                            logger.writer.info("Step {}: Average Loss = {}".format(step, sum(tr_loss) / len(tr_loss)))
                            tr_loss = []
                        
                    if step % args.eval_interval == 0:                              
                        if args.do_test:
                            with torch.no_grad():
                                score2, candidate2 = evaluate_bi_model(bi_model, test_dataset, args, 
                                    save_dir=args.entity_save_dir, 
                                    mode="test", return_all_score=False, logger=logger)
                                if is_master: 
                                    logger.writer.info("hit {} in test set".format(score2))
                                    logger.painter.add_scalar('test_score',score2,step)
                                #! for train
                                score3, candidates3 = evaluate_bi_model(bi_model, train_dataset, args, 
                                        save_dir=args.entity_save_dir, 
                                        mode="train", return_all_score=False, logger=logger)
                                if is_master: 
                                    logger.writer.info("hit {} in train set".format(score3))
                                    logger.painter.add_scalar('train_score',score3,step)

                        if args.do_eval:
                            with torch.no_grad():
                                score1, candidates1 = evaluate_bi_model(bi_model, valid_dataset, args, 
                                    save_dir=args.entity_save_dir, 
                                    mode="valid", return_all_score=False, logger=logger)
                                if is_master: 
                                    logger.writer.info("hit {} in valid set".format(score1))
                                    logger.painter.add_scalar('valid_score',score1,step)
                                    # #print(score2)
                                    if max_score < score1: #!best
                                        torch.save(bi_model.state_dict(), logger.checkpoint_path)
                                        max_score = score1
                                        best_candiate = candidates1
                                        max_score_train, max_score_test = score3, score2
                                        best_candiate_train, best_candiate_test = candidates3, candidate2
                            
                                
            # # save ckpt each epoch
                # if is_master:
                #     logger.writer.info("start saving ckpt for epoch {}".format(e))
                #     torch.save(bi_model.state_dict(), os.path.join(logger.sub_dir, 'epoch_{}.bin'.format(e)))
        del optimizer
    
    #! No matter train/not train, evaluate on the test_set 
    # if args.do_test:
    #     with torch.no_grad():
    #         score2, candidates = evaluate_bi_model(bi_model, test_dataset, args, 
    #                                 save_dir=args.entity_save_dir, 
    #                                 mode="test", return_all_score=False, logger=logger)
    #         if is_master: 
    #             logger.writer.info("hit {} in test set".format(score2))
    #             # for none master process, return two non
    #             if max_score < score2: #!best
    #                 if args.do_train: # if infer only, do not save
    #                     torch.save(bi_model.state_dict(), logger.checkpoint_path)
    #                 max_score = score2
    #                 best_candiate = candidates

    # save the best candidate
    if is_master:
        with open(os.path.join(logger.sub_dir, "valid_candidate_@{:.2f}".format(max_score)), 'w') as f:
            for candidate in best_candiate:
                f.write(json.dumps(candidate) + '\n')
        with open(os.path.join(logger.sub_dir, "train_candidate_@{:.2f}".format(max_score_train)), 'w') as f:
            for candidate in best_candiate_train:
                f.write(json.dumps(candidate) + '\n')
        with open(os.path.join(logger.sub_dir, "test_candidate_@{:.2f}".format(max_score_test)), 'w') as f:
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

    if args.do_train:
        train_dataset = ZeshelDataset(
            mode='train',
            args=args,
            tokenizer=tokenizer,
        )

    if args.do_eval:
        valid_dataset = ZeshelDataset(
            mode='valid',
            args=args,
            tokenizer=tokenizer,
        )

    if args.do_test:
        test_dataset = ZeshelDataset(
            mode='test',
            args=args,
            tokenizer=tokenizer,
        )
    
    #! Main
    if args.n_gpu <= 1:
        main(0, args, train_dataset, valid_dataset, test_dataset, tokenizer,)
    else:
        mp.spawn(main, args=(args, train_dataset, valid_dataset, test_dataset, tokenizer,), nprocs=args.n_gpu, join=True)
    #main(0, args, train_dataset, valid_dataset, test_dataset, tokenizer,)