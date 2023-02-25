# from ast import Sub
from transformers import BertTokenizerFast
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import lazy_groups_of

from time import time
from tqdm import trange
import tqdm

def extract_kg_batch(text_ids, tokenizer, predictor, print_info=False):
    text_all = [{"sentence" :tokenizer.decode(text_id)} for text_id in text_ids]
    res_all = predictor.predict_batch_json(text_all)
    return list(zip(*[analysis(res, tokenizer, print_info=print_info) for res in res_all]))

def extract_kg(text_id, tokenizer, predictor, print_info=False):
    
    text = tokenizer.decode(text_id)

    res = predictor.predict(text)

    return analysis(res,tokenizer, print_info=print_info)

def analysis(res, tokenizer, print_info=False):
    word_span_to_node = {} #! dict, (start word, end word): node_idx
    rels = [] #! adj information

    # extract verb
    for verb_info in res["verbs"]:
        all_tags = verb_info['tags']

        Subject = []
        Verb = []
        Object = []

        start = 0
        pre_tag = all_tags[0]
        for idx in range(1,len(all_tags)+1):
            if idx==len(all_tags) or all_tags[idx][1:] != pre_tag[1:]:
                if pre_tag!= 'O':
                    pos = (start, idx-1)
                    # res["words"][start:idx] ---> tag
                    # print(res["words"][start:idx], all_tags[start:idx], pre_tag)
                    if "-V" in pre_tag:
                        Verb.append(pos) # Verb
                    elif "-ARG0" in pre_tag:
                        Subject.append(pos)
                    elif "-ARG2" in pre_tag or "-ARG1" in pre_tag:
                        Object.append(pos)
                    elif len(Verb) == 0:
                       Subject.append(pos)
                    else:
                       Object.append(pos)
                start = idx
            if idx < len(all_tags):
                pre_tag = all_tags[idx]

        # Check for Triplet
        if len(Subject)==0 or len(Verb)==0 or len(Object) ==0:
            continue
        #assert len(Verb)==1, print("Not One Verb",Verb,all_tags)

        # Add information
        if word_span_to_node.get(Verb[0],None)==None:
            word_span_to_node[Verb[0]] = len(word_span_to_node) + 1 #! count from 1
        for s in Subject:
            for o in Object:
                if word_span_to_node.get(s,None)==None:
                    word_span_to_node[s] = len(word_span_to_node) + 1
                if word_span_to_node.get(o,None)==None:
                    word_span_to_node[o] = len(word_span_to_node) + 1   
                rels.append([word_span_to_node[s],word_span_to_node[Verb[0]],word_span_to_node[o]])

    
    node_to_word_span = {v:k for k,v in word_span_to_node.items()}
    if print_info:
        print("Node num: ",len(word_span_to_node), "Rel num: ", len(rels))
        for r in rels:
            ss = res["words"][node_to_word_span[r[0]][0]:node_to_word_span[r[0]][1]+1]
            vv = res["words"][node_to_word_span[r[1]][0]:node_to_word_span[r[1]][1]+1]
            oo = res["words"][node_to_word_span[r[2]][0]:node_to_word_span[r[2]][1]+1]
            print("{} \t {} \t {}".format(ss,vv,oo))
    
    # Get text ids and corresponding node_mask&rels
    text_ids = []
    word_pos = []
    pos = 0
    for word in res["words"]:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        text_ids += ids
        word_pos.append([pos,pos+len(ids)-1])
        pos += len(ids)
    #assert len(text_ids)==len(text_id), print("Wrong length", text_ids, text_id)
    node_mask = [[word_pos[v[0]][0],word_pos[v[1]][1]] for k,v in node_to_word_span.items()]
    
    
    return text_ids, node_mask, rels

if __name__=="__main__":
    text = "they lived at 5 crimea street until their house was blown up when the army detonated an unexploded bomb found in the yard ."
    # text = "iris rigby [unused2] iris rigby ( nee nuttall ) was the wife of jackie rigby and the only daughter of alf . \
    # iris , a kind , gentle girl , was killed in the blitz in 1940 which left jackie a widower in his thirties . \
    # they lived at 5 crimea street until their house was blown up when the army detonated an unexploded bomb found in the yard . \
    # jackie and father - in - law alf moved into 9 coronation street in april 1946 . \
    # jackie and alf did a moonlight flit in may 1950 , owing three weeks ' rent ."

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",cache_dir="/apdcephfs/private_takiwu/Ger/model_cash")
    # https://github.com/allenai/allennlp/issues/2102 use cuda rather than multiprocess to speed up
    # https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz download from here
    predictor = Predictor.from_path("/apdcephfs/private_takiwu/Ger/model_cash/openie-model.2020.03.26.tar.gz", cuda_device=0)
    text_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    
    t0 = time()
    for _ in trange(100):
        extract_kg(text_id[:128], tokenizer, predictor, print_info=False)
    print(time()-t0)

