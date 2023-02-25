
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import lazy_groups_of
from time import time


#predictor = Predictor.from_path("model_cash/openie-model.2020.03.26.tar.gz")#, cuda_device=0)

# text = "iris rigby [unused2] iris rigby ( nee nuttall ) was the wife of jackie rigby and the only daughter of alf . \
#     iris , a kind , gentle girl , was killed in the blitz in 1940 which left jackie a widower in his thirties . \
#     they lived at 5 crimea street until their house was blown up when the army detonated an unexploded bomb found in the yard . \
#     jackie and father - in - law alf moved into 9 coronation street in april 1946 . \
#     jackie and alf did a moonlight flit in may 1950 , owing three weeks ' rent ."

# predictor.predict_json({"sentence": text})

# t0 = time()
# for _ in range(1):
#     predictor.predict_json({"sentence": text})
# print(time()-t0)

# all = [{"sentence": text} for _ in range(400)]
# t0 = time()
# for batch_sample in lazy_groups_of(all,400):
#     predictor.predict_batch_json(batch_sample)
# print(time()-t0)

#--------------------------------------------#

#from data_loader import EncodeDataset
#from transformers import BertTokenizerFast

#tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",cache_dir="model_cash")
#predictor = Predictor.from_path("model_cash/openie-model.2020.03.26.tar.gz", cuda_device=0)

# EncodeDataset(document_path="data/zeshel/documents",
#     world="star_trek", 
#     tokenizer=tokenizer,
#     max_seq_len=128, 
#     predictor=predictor, 
#     debug=False, 
#     cand_batch_size=250,
#     )

# EncodeDataset(document_path="data/zeshel/documents",
#     world="ice_hockey", 
#     tokenizer=tokenizer,
#     max_seq_len=128, 
#     predictor=predictor, 
#     debug=False, 
#     cand_batch_size=250,
#     )
# A = [[],[[1,2,3],[2,3,4]], [[1,2,3]]]
# def get_len(x):
#     if not x: return 0
#     return len(x)
# length = max([get_len(x) for x in A])
# print([x+[[-1]*3]*(length-get_len(x)) for x in A])
#----------------------------------#
# import torch
# import os
# import json

# P = "data/zeshel/documents/preprocess"
# P1 = "data/zeshel/documents"

# WORLDS = {
#     'train': [("american_football", 31929), ("doctor_who", 40281), ("fallout", 16992), ("final_fantasy", 14044), ("military", 104520), ("pro_wrestling", 10133), ("starwars", 87056), ("world_of_warcraft", 27677)],
#     'valid': [("coronation_street", 17809), ("muppets", 21344), ("ice_hockey", 28684), ("elder_scrolls", 21712)],
#     'test': [("forgotten_realms", 15603), ("lego", 10076), ("star_trek", 34430), ("yugioh", 10031)]
# }

# for k,v in WORLDS.items():
#     for pair in v:
#         samples = torch.load(os.path.join(P,"{}.pt".format(pair[0])))
#         assert len(samples)== pair[1], print(pair)
#         D = os.path.join(P1, "{}.json".format(pair[0]))
#         title_all = []
#         with open(D, 'r') as f:
#             for line in f:
#                 info = json.loads(line)
#                 title_all.append(info["title"])
#         samples_new = []
#         for idx,sample in enumerate(samples):
#             sample["title"] = title_all[idx]
#             samples_new.append(sample)
#         with open(os.path.join(P, "{}.jsonl".format(pair[0])), 'w') as f:
#             for sample in samples_new:
#                 f.write(json.dumps(sample) + '\n')
#         print(1)

# print(1)
#-------------------------------------#
import os
import json

A = "data/zeshel/documents/preprocess"
WORLDS = {
    'train': [("american_football", 31929), ("doctor_who", 40281), ("fallout", 16992), ("final_fantasy", 14044), ("military", 104520), ("pro_wrestling", 10133), ("starwars", 87056), ("world_of_warcraft", 27677)],
    'valid': [("coronation_street", 17809), ("muppets", 21344), ("ice_hockey", 28684), ("elder_scrolls", 21712)],
    'test': [("forgotten_realms", 15603), ("lego", 10076), ("star_trek", 34430), ("yugioh", 10031)]
}
for k,v in WORLDS.items():
    for pair in v:
        name = pair[0]
        p = os.path.join(A,"{}.jsonl".format(name))
        samples = []
        with open(p, 'r') as f:
            for line in f:
                info = json.loads(line)
                info["token_ids"] = info["token_ids"][:-1] + [102]
                samples.append(info)
        
        p1 = os.path.join("data/zeshel/documents/preprocess1","{}.jsonl".format(name))
        with open(p1, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

