#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import os
os.environ["WANDB_MODE"] = "offline"
import torch
import sys
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from tqdm import tqdm
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from utils.perf import print_throughput
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--init_global_step", type=int, default=0, help="")
    parser.add_argument("--part_data_size", type=int, default=-1, help="")
    parser.add_argument('--delete_first_token', action='store_true')
    parser.add_argument("--loss_scale", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--loss_scale_window", type=float, default=1000, help="Weight decay to use.")
    parser.add_argument('--force_max_seq_len', action='store_true')
    parser.add_argument('--project_name',
                        type=str,
                        default='',
                        )
    parser.add_argument('--experiment_name',
                        type=str,
                        default='',
                        )
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/share/project/weiyifan/KG_RAG/results',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--save_interval",
                        type=int,
                        default=100,
                        )
    parser.add_argument("--eval_interval", type=int, default=0, help="")
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def save_model(model, tokenizer, global_step, args, epoch=None):
    model = convert_lora_to_linear_layer(model)
    if epoch is None:
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, str(global_step))
        if args.zero_stage == 3:
            save_zero_three_model(model, args.global_rank, os.path.join(args.output_dir, str(global_step)), args.zero_stage)
    else:
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f'epoch-{epoch}-'+str(global_step))
        if args.zero_stage == 3:
            save_zero_three_model(model, args.global_rank, os.path.join(args.output_dir, f'epoch-{epoch}-'+str(global_step)), args.zero_stage)

class TextDataset_V2(Dataset):
    def __init__(self, file_path, tokenizer, args):
        lines = open(file_path).readlines()
        if args.part_data_size > 0:
            lines = lines[:args.part_data_size]
        self.sessions = [json.loads(line) for line in lines if line.strip() != '']
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.sessions)

    def encode(self, text: str):
        if self.args.delete_first_token:
            return self.tokenizer.encode(text)[1:]
        else:
            return self.tokenizer.encode(text)

    def __getitem__(self, idx):
        tokens = []
        masks = []
        messages = self.sessions[idx]["content"]
        if 'qwen' in self.args.model_name_or_path.lower():
            tkns = self.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
            for midx, message in enumerate(messages):
                if message['role'].lower() in ['user', 'human']:
                    tkns = self.encode(f"<|im_start|>user\n{message['content']}<|im_end|>\n")
                    tokens += tkns
                    masks += [0] * len(tkns)
                else:
                    tkns = self.encode('<|im_start|>assistant\n')
                    tokens += tkns
                    masks += [0] * len(tkns)
                    tkns = self.encode(message["content"]+'<|im_end|>\n')+[self.tokenizer.eos_token_id] + self.encode('\n')
                    tokens += tkns
                    masks += [1] * len(tkns)
        elif 'llama' in self.args.model_name_or_path.lower():
            tkns = self.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>")
            for midx, message in enumerate(messages):
                if message['role'].lower() in ['user', 'human']:
                    tkns = self.encode(f'<|start_header_id|>user|end_header_id|>\n\n{message["content"]}<|eot_id|>')
                    tokens += tkns
                    masks += [0] * len(tkns)
                else:
                    tkns = self.encode('<|start_header_id|>assistant<|end_header_id|>\n\n')
                    tokens += tkns
                    masks += [0] * len(tkns)
                    tkns = self.encode(message["content"]+"<|eot_id|>")+[self.tokenizer.eos_token_id] + self.encode('\n')
                    tokens += tkns
                    masks += [1] * len(tkns)
        elif 'baichuan' in self.args.model_name_or_path.lower():
            for midx, message in enumerate(messages):
                if message['role'].lower() in ['user', 'human']:
                    tkns = self.encode(f'<|start_header_id|>user|end_header_id|>\n\n{message["content"]}<|eot_id|>')
                    tokens += tkns
                    masks += [0] * len(tkns)
                else:
                    tkns = self.encode('<|start_header_id|>assistant<|end_header_id|>\n\n')
                    tokens += tkns
                    masks += [0] * len(tkns)
                    tkns = self.encode(message["content"]+"<|eot_id|>")+[self.tokenizer.eos_token_id] + self.encode('\n')
                    tokens += tkns
                    masks += [1] * len(tkns)
        elif "mistral-7b" in self.args.model_name_or_path.lower():
            for midx, message in enumerate(messages):
                if message['role'].lower() in ['user', 'human']:
                    tkns = self.encode(f'<s>[INST] {message["value"]} [/INST] ')
                    tokens += tkns
                    masks += [0] * len(tkns)
                else:
                    # tkns = self.encode('<|start_header_id|>assistant<|end_header_id|>\n\n')
                    # tokens += tkns
                    # masks += [0] * len(tkns)
                    tkns = self.encode(message["value"]+"</s>")+[self.tokenizer.eos_token_id] + self.encode('\n')
                    tokens += tkns
                    masks += [1] * len(tkns)
        else:
            print('Error!!!!!!!!!!')


        return {
            "token_ids": tokens,
            "loss_mask": masks
        }

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    project_name = args.project_name
    experiment_name = args.experiment_name
    run_dir = os.path.join(args.data_output_path, 'log',  experiment_name)

    args.global_rank = torch.distributed.get_rank()
    print_rank_0("***** Running wandb *****", args.global_rank)
    wandb.login(key="")
    if args.global_rank == 0:
        wandb.init(config=args,
               entity='',
               project=project_name,
               name=experiment_name,
               dir=run_dir,
               job_type="training",
               )

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    nll_loss = nn.NLLLoss(reduce=False).to(args.local_rank)

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    print_rank_0("***** Running load model *****", args.global_rank)
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    def pad_collate_v2(batch):
        new_batch, mask_batch, prompt_lens = [], [], []
        for item in batch:
            input_ids = item['token_ids']
            mask_ids = item['loss_mask']
            assert len(input_ids) == len(mask_ids)
            
            input_ids = input_ids[-args.max_seq_len:]
            mask_ids = mask_ids[-args.max_seq_len:]
            seq_len = len(input_ids)
                
            prompt_lens.append(seq_len)
            mask_batch.append(mask_ids)
            new_batch.append(input_ids)

        real_max_len = max(prompt_lens)

        pad_id = tokenizer.eos_token_id

        if args.force_max_seq_len:
            new_batch = [item + [pad_id]*(args.max_seq_len-len(item)) for item in new_batch]
            mask_batch = [item + [0]*(args.max_seq_len-len(item)) for item in mask_batch]
        else:
            new_batch = [item + [pad_id]*(real_max_len-len(item)) for item in new_batch]
            mask_batch = [item + [0]*(real_max_len-len(item)) for item in mask_batch]
        return torch.LongTensor(new_batch), torch.LongTensor(mask_batch), None

    def cal_loss_v2(input, target, masks):

        """Input: [batch, length, vocab]; target: [batch, length]; seq_lens: [batch]"""
        log_probs = F.log_softmax(input, dim=-1).transpose(1, 2)  # 类别维度要放在dim=1的位置，nn.NLLLoss的要求
        loss = nll_loss(log_probs, target)
        masked_loss = loss * masks
        ## 增加pad部分的loss，让模型学到终止条件
        mean_loss = torch.sum(masked_loss) / torch.sum(masks)
        # pad_loss = torch.sum(loss * pad_mask) / torch.sum(pad_mask)
        return mean_loss
    
    dataset_class = TextDataset_V2
    pad_func = pad_collate_v2
    print_rank_0("***** Running data process *****", args.global_rank)
    train_dataset = dataset_class(args.data_path[0], tokenizer, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,  sampler=train_sampler, collate_fn=pad_func)
    if args.eval:
        val_dataset = dataset_class(args.data_path[0], tokenizer, args, 'val')
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size,  sampler=val_sampler, collate_fn=pad_func)
    print_rank_0("*****  Data process end*****", args.global_rank)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_warmup_steps = int(0.1 * args.num_train_epochs * num_update_steps_per_epoch) 

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay, args.lora_learning_rate)
    torch.distributed.barrier()
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))
                            
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps= args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)


    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def evaluate(model, val_loader):
        loss_list = []
        for batch_id, batch_data in enumerate(tqdm(val_loader, desc=f'Validation', disable=args.global_rank!=0)):
            val_loader.sampler.set_epoch(0)

            inputs, masks, _ = batch_data
            inputs = inputs.to(args.local_rank)
            masks = masks.to(args.local_rank)

            outputs = model(inputs, use_cache=False)
            logits = outputs.logits
            loss = cal_loss_v2(logits[:, :-1, :], inputs[:, 1:], masks[:, 1:])

            dist.reduce(loss, 0)
            loss = loss / dist.get_world_size()
            loss_list.append(loss.item())
        mean_val_loss = np.mean(loss_list)
        return mean_val_loss 

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
  
    loss_stack = []
    global_step = args.init_global_step
    model.train()
    for epoch in range(args.num_train_epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        for batch_id, batch_data in enumerate(tqdm(train_dataloader, desc=f'EPOCH:[{epoch+1}/{args.num_train_epochs}]', disable=args.global_rank!=0)):
            
            inputs, masks, _ = batch_data
            if args.global_rank <= 0:
                print(f'Shape: [{list(inputs.shape)}]')
            inputs = inputs.to(args.local_rank)
            masks = masks.to(args.local_rank)

            outputs = model(inputs, use_cache=False)
            logits = outputs.logits
            loss = cal_loss_v2(logits[:, :-1, :], inputs[:, 1:], masks[:, 1:])
            model.backward(loss)
            
            if args.global_rank == 0:
                wandb.log({
                        "optimizer._global_grad_norm": model.optimizer._global_grad_norm, 
                        "loss_scale": model.optimizer.cur_scale
                        }, step=global_step)
           
            model.step()

            dist.reduce(loss, 0)
            loss = loss / dist.get_world_size()
            loss_stack.insert(0, loss.item())

            if args.global_rank == 0:
                wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']}, step=global_step)

            loss_stack = loss_stack[:20]
            mean_loss = np.mean(loss_stack)

            if args.global_rank <= 0:
                print(f'Epoch[{epoch}] Batch Step[{batch_id}/{len(train_dataloader)}] Global Step[{global_step}] Loss:[{loss.item()}] Mean_loss: [{mean_loss}] Shape: [{list(inputs.shape)}]')
                curr_lr = optimizer.param_groups[0]['lr']
            

            # in-epoch save and eval
            if batch_id != len(train_dataloader) -1:
                if args.save_interval > 0 and (global_step+1) % args.save_interval == 0:
                    save_model(model, tokenizer, global_step, args)
                if args.eval and args.eval_interval > 0 and (global_step+1) % args.eval_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        val_loss = evaluate(model, val_dataloader)
                        if args.global_rank == 0:
                            wandb.log({"val_loss": val_loss}, step=global_step)
                    model.train()
            global_step += 1
            
        model.tput_timer.update_epoch_count()
        
        if args.eval:
            model.eval()
            with torch.no_grad():
                val_loss = evaluate(model, val_dataloader)
                if args.global_rank == 0:
                    wandb.log({"val_loss": val_loss}, step=global_step)
        # save model
        save_model(model, tokenizer, global_step, args, epoch=epoch)

    
if __name__ == "__main__":
    main()
