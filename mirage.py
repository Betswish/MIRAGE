import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from nltk import sent_tokenize
import re
import numpy as np
import string
import torch
import yaml

import pandas as pd
from transformers import AutoTokenizer
from utils import *
import inseq
from inseq.commands.attribute_context.attribute_context import AttributeContextArgs, attribute_context, attribute_context_with_model


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def mirage_cite(res_mirage, cti_threshold, start_pos_sent, end_pos_sent, topk_CCI, doc_seps):
    res = []

    sum_weight = 0
    sum_value = np.zeros(len(res_mirage['input_context_tokens']))
    
    for i in res_mirage['cci_scores']:
        # CTI Filtering
        if not (i["cti_idx"] >= start_pos_sent and i["cti_idx"] < end_pos_sent): continue
        if i['cti_score'] >= cti_threshold:
            # CCI Focus
            CCI_value = np.array(i['input_context_scores'])
            if topk_CCI == 0:
                cci_threshold = np.mean(CCI_value)
            elif topk_CCI < 0:
                cci_threshold = (1+topk_CCI/100) * np.max(CCI_value) - topk_CCI/100 * np.min(CCI_value)
            else:
                cci_threshold = np.sort(CCI_value)[-topk_CCI]
            zero_idx = CCI_value < cci_threshold
            CCI_value[zero_idx] = 0

            sum_value += CCI_value

        if i['cti_score'] < cti_threshold: break

    sum_tmp = 0
    for i, v in enumerate(sum_value):
        sum_tmp += v
        if doc_seps[i] or (i == len(sum_value)-1): # meet '\n'
            res.append(sum_tmp)
            sum_tmp = 0
    return res

def generate_answer(prompt, model, tokenizer, max_tokens, temperature, top_p):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    stop = []
    stop_token_ids = list(set([tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [model.config.eos_token_id]))
    if tokenizer.unk_token_id in stop_token_ids:
        stop_token_ids.remove(tokenizer.unk_token_id) 
    outputs = model.generate(
            **inputs,
            do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=stop_token_ids
    )
    generation = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return generation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Input data file")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--CTI", type=int, default=1, help="CTI filtering strategy: How many standard deviations over average")
    parser.add_argument("--CCI", type=int, default=-5, help="CCI filtering strategy: Top k if k > 0; Top (-k)% if k < 0")

    parser.add_argument("--seed", type=int, default=42, help="Seed for random stuffs")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")

    parser.add_argument("--f_with_ans", action="store_true", help="Whether input data file already has LLM generations.")
    parser.add_argument("--only_cite", action="store_true", help="Only re-generate citations with new CTI and CCI thresholds")
    
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.only_cite:
        assert args.f_with_ans, "--only_cite can only used when the input data contains the LLM outputs, namely setting --f_with_ans"
    
    np.random.seed(args.seed)

    # CTI and CCI parameters
    topk_CTI = args.CTI
    #topk_CTI = 1 # 1 means over average+1SD
    #topk_CTI = 0 # 0 means over average

    topk_CCI = args.CCI
    #topk_CCI = -5 # -5 means range top5%
    #topk_CCI = 3 # 3 means top 3
    #topk_CCI = 0 # 0 means average (not used)
    
    cite_idx_acs = False # whether MIRAGE citations in ascending order

    model, tokenizer = load_model(args.model)
    data = json.load(open(args.f))

    if not args.f_with_ans:
        prefix = args.model.lower().replace('/','_') + "-" + args.f..split("/")[-1].split(".")[0] + "-" + args.config..split("/")[-1].split(".")[0] + '-seed' + str(args.seed)
    else:
        prefix = args.f..split("/")[-1].split(".")[0]
    # First, generate and save LLM generation
    # If already have LLM generation
    if args.f_with_ans:
        for idx, item in enumerate(tqdm(data)):
            item['output'] = item['output'].strip()
            for i in range(10):
                r_tmp = "\n" * (10-i)
                item['output'] = item['output'].replace(r_tmp, " ")
    else:
        for idx, item in enumerate(tqdm(data)):
            doc_list = item['docs']
            input_context_text = "".join([make_doc_prompt(doc, doc_id, args.doc_prompt, use_shorter=None) for doc_id, doc in enumerate(doc_list)])
            input_current_text = item['question']
            input_template = args.demo_prompt.replace("{INST}", args.instruction).replace("{Q}", "{current}").replace("{A}</s>", "").replace("{A}", "").replace("{D}", "{context}").rstrip()
        
            prompt = input_template.replace("{current}", input_current_text).replace("{context}", input_context_text)
            prompt_len = len(tokenizer.tokenize(prompt))
            item['output'] = generate_answer(prompt, model, tokenizer, min(args.max_new_tokens, args.max_length-prompt_len), args.temperature, args.top_p)

            item['output'] = item['output'].strip()
            for i in range(10):
                r_tmp = "\n" * (10-i)
                item['output'] = item['output'].replace(r_tmp, " ")
            
    if not os.path.exists("data_input_with_ans"):
        os.makedirs("data_input_with_ans")
    json.dump(data, open("data_input_with_ans/" + prefix + ".json", "w"), indent=4)


    # Second, analyze model internals with MIRAGE
    save_dir_mirage = './internal_res/'
    if not os.path.exists(save_dir_mirage):
        os.makedirs(save_dir_mirage)

    if not args.only_cite:
        # Load model
        model_mirage = inseq.load_model(
                model,
                "saliency",
                model_kwargs={"device_map": 'cuda:0', "torch_dtype": torch.float16},
                tokenizer_kwargs={"use_fast": False},
        )

        stop = []
        stop_token_ids = list(set([tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [model.config.eos_token_id]))
        if tokenizer.unk_token_id in stop_token_ids:
            stop_token_ids.remove(tokenizer.unk_token_id)

        decoder_input_output_separator = ' ' 
        special_tokens_to_keep = []
        if "zephyr" in args.model.lower():
            decoder_input_output_separator = '\n '
            special_tokens_to_keep = ["</s>"]

        num_empty = 0
        for idx, item in enumerate(tqdm(data)):
            if item["output"] == "": 
                num_empty += 1
                continue
            doc_list = item['docs']
            input_context_text = "".join([make_doc_prompt(doc, doc_id, args.doc_prompt, use_shorter=None) for doc_id, doc in enumerate(doc_list)])
            input_current_text = item['question']
            input_template = args.demo_prompt.replace("{INST}", args.instruction).replace("{Q}", "{current}").replace("{A}</s>", "").replace("{A}", "").replace("{D}", "{context}").rstrip()
            contextless_input_current_text = input_template.replace("{context}", "")
            output_current_text = item["output"]
            
            save_path = save_dir_mirage + prefix + '-' + str(idx) + '.json'
            lm_rag_prompting_example = AttributeContextArgs(
                    model_name_or_path=args.model,
                    input_context_text=input_context_text,
                    input_current_text=input_current_text,
                    output_template="{current}",
                    input_template=input_template,
                    contextless_input_current_text=contextless_input_current_text,
                    show_intermediate_outputs=False,
                    attributed_fn="contrast_prob_diff",
                    context_sensitivity_std_threshold=0,
                    output_current_text=output_current_text,
                    attribution_method="saliency",
                    attribution_kwargs={"logprob": True},
                    save_path=save_path,
                    tokenizer_kwargs={"use_fast": False},
                    model_kwargs={
                        "device_map": 'auto',
                        "torch_dtype": torch.float16,
                        "max_memory": get_max_memory(),
                        "load_in_8bit": False,
                        },
                    generation_kwargs={
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "num_return_sequences": 1,
                        "eos_token_id": stop_token_ids
                        },
                    decoder_input_output_separator=decoder_input_output_separator,
                    special_tokens_to_keep=special_tokens_to_keep,
                    show_viz=False,
                    )

            gen = attribute_context_with_model(lm_rag_prompting_example, model_mirage)
            
            #print(gen)

    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.padding_side = "left"
    
    new_data = []
    num_empty = 0
    for idx, item in enumerate(tqdm(data)):
        if item["output"] == "": 
            new_data.append(item)
            num_empty += 1
            continue
    
        item["output"] = item["output"].strip()
        output = item["output"]

        # read MIRAGE json results
        read_path = save_dir_mirage + prefix + '-'+str(idx)+'.json'
        with open(read_path) as r:
            res_mirage = json.load(r)

        if topk_CTI >= 0:
            cti_threshold = np.mean(res_mirage["cti_scores"]) + topk_CTI * np.std(res_mirage["cti_scores"])
        else:
            raise ValueError('CTI filtering parameter should be equal or larger than 0.')

        sents = sent_tokenize(output)
        # check num and index of '\n' in the retrieved docs (i.e. <0x0A> in Llama, zephyr, mistral)
        # e.g. num should constantly be 5 on ELI5
        doc_seps = np.array(res_mirage["input_context_tokens"])
        doc_seps = doc_seps == '<0x0A>'
        num_doc = pd.value_counts(res_mirage["input_context_tokens"])["<0x0A>"]
        
        new_output = ""
        start_pos_sent = 0
        end_pos_sent = 0
        for sent in sents:
            # e.g. original citation index: [1,3,4]
            original_ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] 
            end_pos_sent = start_pos_sent + len(tokenizer.tokenize(sent))
            
            # e.g. Filtered CCI values for each doc, e.g. [0, 0, 20, 3, 0]; always length == num_doc
            cite_result_mirage = mirage_cite(res_mirage, cti_threshold, start_pos_sent, end_pos_sent, topk_CCI, doc_seps)
            start_pos_sent = end_pos_sent

            if len(cite_result_mirage) >= 0:
                #print("\n-----")
                sent = remove_citations(sent)
               
                best_doc_id_tmp = {i: v for i, v in enumerate(cite_result_mirage) if v}
                best_doc_id = list(dict(sorted(best_doc_id_tmp.items(), key=lambda item: item[1], reverse=True)).keys())
                best_doc_id = best_doc_id[: min(args.at_most_citations, len(best_doc_id))]

                if cite_idx_acs:
                    best_doc_id = sorted(best_doc_id)

                best_doc_id_str = ""
                for i in best_doc_id:
                    best_doc_id_str += "[" + str(i+1) + "]"
                sent = best_doc_id_str + " " + sent
            
            new_output += sent + " "

        item['output'] = new_output.rstrip().rstrip(",")
        print("\n-----")
        print("Output with MIRAGE AA:" + item['output'])
        new_data.append(item)

    print("num_empty:")
    print(num_empty)
    print()
    data = new_data 
    
    tag = f".mirage"     
    tag += f"_CTI_{topk_CTI}"
    tag += f"_CCI_{topk_CCI}"

    if cite_idx_acs:
        tag += '_acs'

    json.dump(data, open(args.f + f"{tag}", 'w'), indent=4)

if __name__ == "__main__":
    main()
