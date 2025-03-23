# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        
        if dataset_name == 'SFT/v916':
            self.raw_datasets = load_dataset("json", data_files={'train': '/home/baaidial/zhaohanyu/Data/v916/output2/train.jsonl', 'test': '/home/baaidial/zhaohanyu/Data/v916/output2/test.jsonl'})
        if dataset_name == 'SFT/v925':
            self.raw_datasets = load_dataset("json", data_files={'train': '/home/baaidial/zhaohanyu/Data/v925/output/train.jsonl', 'test': '/home/baaidial/zhaohanyu/Data/v925/output/test.jsonl'})
        if dataset_name == 'PPO2/v1':
            self.raw_datasets = load_dataset("json", data_files={'train': "/share/zhaohanyu/Data/PPO/v1/train.jsonl", 'test':"/share/zhaohanyu/Data/PPO/v1/test.jsonl"})
        if dataset_name == 'RM2/v3':
            self.raw_datasets = load_dataset("json", data_files={'train': "/share/zhaohanyu/Data/RM/v3/train.jsonl", 'test':"/share/zhaohanyu/Data/RM/v3/test.jsonl"})
        
    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


class SFT921Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "SFT/v921"
        self.dataset_name_clean = "SFT_v921"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen'] # 数据集中已经拼接<|startofpiece|> <|endofpiece|> 
        # return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class SFT925Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "SFT/v925"
        self.dataset_name_clean = "SFT_v925"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen'] # 数据集中已经拼接<|startofpiece|> <|endofpiece|> 
        # return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class SFT916Dataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "SFT/v916"
        self.dataset_name_clean = "SFT_v916"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class SFTBelloDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "SFT/belle"
        self.dataset_name_clean = "SFT_belle"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class PPOV4Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "PPO/v4"
        self.dataset_name_clean = "PPO_v4"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        # return "Human: " +  sample['input'] +' Assistant: ' 
        return "<|startofpiece|> " + sample['input']+ " <|endofpiece|> " 

class RM2V4Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "RM2/v4"
        self.dataset_name_clean = "RM2_v4"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['rejected']

class RM2V3Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "RM2/v3"
        self.dataset_name_clean = "RM2_v3"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['rejected']
  
class RM2V2Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "RM2/v2"
        self.dataset_name_clean = "RM2_v2"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return "<|startofpiece|> " + sample['prompt'] + " <|endofpiece|> "+ sample['rejected']
    
class PPO2v1Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "PPO2/v1"
        self.dataset_name_clean = "PPO2_v1"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        # return "Human: " +  sample['input'] +' Assistant: ' 
        return "<|startofpiece|> " + sample['input']+ " <|endofpiece|> " 
