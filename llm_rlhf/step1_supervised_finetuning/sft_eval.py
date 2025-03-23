# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import re
import json
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM, AutoTokenizer)


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from utils.model.model_utils import create_hf_model

# from utils.ds_utils import get_train_ds_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to baseline model",
        
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to baseline model",
    
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        default='/home/zhaohy/llm_rlhf/step1_supervised_finetuning/checkpoints2/qwen1.5_7B_sft_eevol_v5_detia_ability_deficient_v2/epoch-2-14532',
        # required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default='/share/projset/transfer_0422/Qwen1.5-7B-Base',
        help="Path to pretrained model",
        # required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="Chinese",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=20):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample, eos_token_id=tokenizer.eos_token_id,
                                  num_return_sequences=num_return_sequences,#, temperature=0.7,
                                  max_new_tokens=max_new_tokens)
    # print(generate_ids)
    # print('++++++++++++++++++\n')

    result = tokenizer.batch_decode(generate_ids,
                                    # skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result

def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result

def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()

def prompt_eval(args, model_fintuned, tokenizer, device):
   
    # while True:
    with open('output/aplaca.jsonl', 'w') as w:
        with open('/home/zhaohy/Alpaca/alpacaeval/alpaca_eval.json') as f:
            for item in tqdm(json.load(f)):
                prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + item["instruction"] + '<|im_end|>\n<|im_start|>assistant\n'
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                r_finetune_g = generate(model_fintuned,
                                            tokenizer,
                                            inputs,
                                            num_beams=1, #3
                                            do_sample = False,#Trues
                                            num_return_sequences=args.num_return_sequences,
                                            max_new_tokens=args.max_new_tokens)
                output = r_finetune_g[0][len(prompt):]
                output = re.sub(r"<\|im_end\|>\n<\|endoftext\|>",'',output)
                dic = {}
                dic["dataset"] = item["dataset"]
                dic["instruction"] = item["instruction"]
                dic["output"] = output
                dic["generator"] = item["generator"]
                print(dic)
                print('+++++++++++++++++++++++++++++++++++++++\n')
                w.write(json.dumps(dic, ensure_ascii=False)+'\n')
                
       

def get_tokenizer(model_name_or_path, fast_tokenizer=True):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,  trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    print(model_name_or_path)
    
    # Locally tokenizer loading has some issue, so we need to force download
    
    tokenizer = get_tokenizer(model_name_or_path, fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    return tokenizer

def main():
    args = parse_args()
    print(args.model_name_or_path_finetune)
    device = torch.device("cuda")

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.tokenizer,
                                #   fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path_finetune,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    

    prompt_eval(args, model, tokenizer, device)

if __name__ == "__main__":
    main()
