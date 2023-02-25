import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, DataLoader

from tqdm import tqdm 
import os

from prettytable import PrettyTable 

import sys

from ger.data_loader import entity_collate_fn, WORLDS, cross_collate_fn

def pretty_visualize(scores, top_k, logger=None):
    rows = []
    for world, score in scores.items():
        rows.append([world] + [round(s * 1.0 / score[1], 4) for s in score[0]])
    
    table = PrettyTable()
    table.field_names = ["World"] + ["R@{}".format(k) for k in top_k]
    table.add_rows(rows)
    print(table)
    if logger:
        logger.writer.info("scores on dataset: \n {}".format(table))


def evaluate_bi_model(model, dataset, args, save_dir, mode, return_all_score, logger):
    model.eval()

    is_master = args.local_rank in [-1,0]

    # for ddp model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        test_module = model.module
    else:
        test_module = model
    
    # encode candidate for each world
    world_entity_pool, world_entity_titles = {}, {}
    for world, world_dataset in dataset.entity_desc.items():
        entity_pool, entity_title = [], []
        if args.n_gpu > 1:
            sampler = DistributedSampler(world_dataset)
        else:
            sampler = SequentialSampler(world_dataset)
        encode_dataloader = DataLoader(dataset = world_dataset, batch_size = args.encode_batch_size, 
            collate_fn=entity_collate_fn, shuffle=False, sampler=sampler)
        
        disable = not is_master
        for sample in tqdm(encode_dataloader, disable=disable):
            candidate_encode = test_module.encode_entity(
                entity = {k:v.cuda(non_blocking=True) for k,v in sample['entity'].items()},
                ).detach().to("cpu") 
            entity_pool.append(candidate_encode)
            entity_title += sample["title"]
            
        world_entity_pool[world] = torch.cat(entity_pool, dim=0)
        world_entity_titles[world] = entity_title
    
    # save the entity encoding
    torch.save([world_entity_pool, world_entity_titles], os.path.join(save_dir,'entity_{}.pt'.format(args.local_rank)))
    print("save!", args.local_rank)

    if args.n_gpu > 1:
        torch.distributed.barrier()
    
    #! 非主进程直接退出
    if not is_master:
        return None, None
    
    # load the encoding into cuda:0
    world_entity_pool, world_entity_titles = {}, {}
    for i in range(args.n_gpu): # load cache by each gpu
        sub_entity_pool, sub_entity_titles = torch.load(os.path.join(save_dir,'entity_{}.pt'.format(i)), map_location='cpu') 
        for world_name, world_num in WORLDS[mode]:
            titles = world_entity_titles.get(world_name, [])
            pool = world_entity_pool.get(world_name, [])
            
            sub_titles = sub_entity_titles[world_name]
            sub_pool = sub_entity_pool[world_name]

            # In SubWorldDistributedSampler we pad some samples; here to drop
            if world_num % args.n_gpu and world_num % args.n_gpu - 1 < i:
                end_idx = len(sub_titles) - 2
                while sub_titles[end_idx] == sub_titles[-1]:
                    end_idx -= 1
                
                sub_titles = sub_titles[:end_idx + 1]
                sub_pool = sub_pool[:end_idx+1, :]
            # After drop, add to whole
            titles += sub_titles
            world_entity_titles[world_name] = titles

            pool.append(sub_pool)
            world_entity_pool[world_name] = pool
    
    # Move to cuda:0
    for key, _ in WORLDS[mode]:
        pool = world_entity_pool[key]
        pool = torch.cat(pool, dim=0).to("cuda:0")
        world_entity_pool[key] = pool 
        #print(world_entity_pool[key].shape)
    
    world_entity_ids_range = {}
    for key, titles in world_entity_titles.items():
        ids_range = {}
        for ids, title in enumerate(titles):
            title_range = ids_range.get(title, [])
            title_range.append(ids)
            ids_range[title] = title_range
        world_entity_ids_range[key] = ids_range # dict, title: [all_title_before]

    #! get top-k
    top_k = [1, 2, 4, 8, 16, 32, 50, 64] # recall@k
    score_metrics = {world_name: [[0] * len(top_k), 0] for world_name, _ in WORLDS[mode]}
    score_metrics['total'] = [[0] * len(top_k), 0]
    candidates = []
    # Then Encode the entities and Compare
    dataloader = DataLoader(dataset=dataset, batch_size=args.eval_batch_size, collate_fn=cross_collate_fn, shuffle=False)
    for batch in tqdm(dataloader): # for mention
        worlds, labels = batch['world'], batch['label_world_idx']
        # get predict
        predict_scores = test_module.score_candidates(
            mention = {k:v.to("cuda:0") for k,v in batch["mention"].items()},
            ctx_world = worlds,
            candidate_pool = world_entity_pool
        ) # batch_size [candidate_num,...,]
         
        for predict_score, world, label in zip(predict_scores, worlds, labels):
            predict_score = torch.softmax(predict_score, dim=-1)
            # rank
            predict_ids = torch.sort(predict_score, -1, descending=True).indices.cpu()
            scores = torch.sort(predict_score, -1, descending=True).values.cpu()
            # get ground truth label
            label_title = dataset.entity_desc[world].get_nth_title(label)
            
            # get different top XX titles
            predict_title = []
            ids = 0
            while len(predict_title) < max(top_k):
                title = world_entity_titles[world][predict_ids[ids]]
                if title not in predict_title:
                    predict_title.append(title)
                ids += 1
            # calculate hit@XX
            for k_idx, k in enumerate(top_k):
                if label_title in predict_title[:k]:
                    score_metrics[world][0][k_idx] += 1
                    score_metrics['total'][0][k_idx] += 1
            score_metrics[world][1] += 1
            score_metrics['total'][1] += 1
        
            candidates.append([{'title': title} for title in predict_title])
    print(score_metrics)
    # table visualize
    pretty_visualize(score_metrics, top_k, logger=logger)

    if return_all_score:
        return score_metrics, candidates
    else:
        #score_metrics['total']: [[recall@1, recall@2, recall@4, recall@8,...], [总数]]
        return score_metrics['total'][0][-1], candidates