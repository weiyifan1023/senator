"""
Run evaluation of medical LLMs on the MedQA, MedMCQA, PubMedQA dataset.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import glob
import re
import torch
from vllm import LLM, SamplingParams
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
import time, argparse
import jsonlines, json
import pandas as pd
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional, Sequence

from benchmark_utils import *
from answer_utils import *

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

one_shot_prompt = """Directly answer the best option in the following format: "Correct answer is [option]. [answer]."
Question: The zygomatic bone does not articulate with:
A. Frontal bone B. Maxillary bone C. Nasal bone D. Temporal bone
Correct answer is C. Nasal bone.
"""
english_prompt = "Directly answer the best option:"
english_prompt_pubmedqa = "Directly answer yes/no/maybe:"

# Declare global variables for tokenizer and model
tokenizer = None
model = None
test_num = 999999


def read_jsonl(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def read_data_file(file_path):
    data_list = []

    if file_path.endswith('.jsonl'):
        with jsonlines.open(file_path) as reader:
            data_list = [obj for obj in reader]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data_list = data if isinstance(data, list) else [data]
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")

    return data_list


def construct_spedical_tokens_dict(tokenizer) -> dict:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    DEFAULT_MASK_TOKEN = "[MASK]"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if tokenizer.mask_token is None or tokenizer.mask_token == '':
        special_tokens_dict["mask_token"] = DEFAULT_MASK_TOKEN

    return special_tokens_dict


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def medqa_eval(args=None):
    """
    Performance Evaluation on the MedQA Benchmark
    :param config or args: the configuration file
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    # data_path = args.data_path
    data_path = "/share/project/weiyifan/KG_RAG/data/benchmark_data/MedQA_USMLE_test.jsonl"
    args.data_path = data_path
    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            question = medqa_format(item)
            prompts.append(question)
            # Get the label answer
            temp_ans = item["answer_idx"]
            answers.append(temp_ans)
            if idx == test_num:
                break

    performance_eval(prompts=prompts, answers=answers, args=args)
    # vllm_performance_eval(prompts=prompts, answers=answers, args=args)


def pubmedqa_eval(args):
    """
    Performance Evaluation on the PubMedQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []
    # data_path = args.data_path
    data_path = "/share/project/weiyifan/KG_RAG/data/benchmark_data/PubMedQA_test.json"
    args.data_path = data_path
    with open(data_path, 'r', encoding='utf8') as file:
        data = json.load(file)
        idx = 0
        # Iterate through each item and print the final decision
        for key, item in data.items():
            question = pubmedqa_format(item)
            prompts.append(question)
            # Get the label answer
            temp_ans = item["final_decision"]
            if temp_ans == "yes":
                temp_ans = 'A'
            elif temp_ans == "no":
                temp_ans = 'B'
            else:
                temp_ans = 'C'
            answers.append(temp_ans)
            idx += 1
            if idx == test_num:
                break

        performance_eval(prompts=prompts, answers=answers, args=args)
        # vllm_performance_eval(prompts=prompts, answers=answers, args=args)


def medmcqa_eval(args):
    """
    Performance Evaluation on the MedMCQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []
    # data_path = args.data_path
    data_path = "/share/project/weiyifan/KG_RAG/data/benchmark_data/train_data/MedMCQA_dev.json"
    args.data_path = data_path
    idx = 0
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            question = medmcqa_format(item)
            prompts.append(question)

            # Get the label answer
            temp_ans = item["cop"]
            if temp_ans == 1:
                temp_ans = 'A'
            elif temp_ans == 2:
                temp_ans = 'B'
            elif temp_ans == 3:
                temp_ans = 'C'
            else:
                temp_ans = 'D'
            answers.append(temp_ans)
            idx += 1
            if idx == test_num:
                break

    performance_eval(prompts=prompts, answers=answers, args=args)
    # vllm_performance_eval(prompts=prompts, answers=answers, args=args)


def usmle_eval(args):
    """
    Performance Evaluation on the Internal USMLE QA (Lindsey, Divya, and Meagan)
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []
    data_path = args.data_path
    with open(data_path, 'r') as f:
        contents = json.loads(f.read())
        for item in contents:
            question = usmle_format(item)

            prompts.append(question)

            # Get the label answer
            temp_ans = item["answer_id"]
            answers.append(temp_ans)

    performance_eval(prompts=prompts, answers=answers, args=args)


def mmlu_eval(args):
    """
    Performance Evaluation on the MMLU Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """

    data_path = "/share/project/weiyifan/KG_RAG/data/benchmark_data/mmlu_english"
    domain_type = "/medicine/" # "/biology/"
    args.data_path = data_path

    prompts = []
    answers = []

    csv_files = glob.glob(data_path + domain_type + "*.csv")
    # 遍历所有csv文件并读取数据
    all_data = []
    for csv_file in csv_files:
        csv_data = pd.read_csv(csv_file, header=None)
        # all_data.append(temp_df)
    # csv_data = pd.concat(all_data, ignore_index=True)
        for index, item in csv_data.iterrows():
            question = mmlu_format(item=item, lang="English")
            prompts.append(question)

            # Get the label answer
            temp_ans = item[5]
            answers.append(temp_ans)

        # Performance evaluation
        # performance_eval(prompts=prompts, answers=answers, args=args)
        vllm_performance_eval(prompts=prompts, answers=answers, args=args)
        print(csv_file)




def vllm_performance_eval(prompts, answers, args):
    start_t = time.time()
    data_name = args.data_path.split("/")[-1].split("_")[0]
    model_name = args.model_name_or_path.split("/")[-1]
    # 使用 os.path.join() 拼接路径
    file_path = os.path.join(args.save_path, f"{data_name}_{model_name}.json")

    # Inference
    responses = []
    sampling_params = SamplingParams(
        top_k=50,
        max_tokens=64,  # 生成的最大 token 数
    )
    outputs = args.llm.generate(prompts, sampling_params)  # 完成推理
    for output in outputs:
        tmp_prompt = output.prompt
        if "mcts" in args.model_name_or_path:
            tmp_gen = "Correct option is " + output.outputs[0].text
        else:
            # tmp_gen = output.outputs[0].text
            tmp_gen = "Correct option is " + output.outputs[0].text
        responses.append(tmp_gen)

    print('Successfully finished generating', len(prompts), 'samples!')

    # compute accuracy metric
    acc = []
    all_out = []  # Save all the model's responses
    invalid_out = []  # Only save the invalid responses
    for idx, (question, response, answer) in enumerate(zip(prompts, responses, answers)):
        if "pubmedqa" in file_path.lower():
            # We separately write codes to evaluate our models' performance
            # on the PubMedQA benchmark (yes/no/maybe)
            prediction = extract_answer_for_pubmedqa(completion=response)
        elif ("mmlu" in file_path.lower() or "medmcqa" in file_path.lower() or "medqa" in file_path.lower()):
            prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
        elif "english_medexpqa" in file_path.lower():
            prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
        elif "english_usmle" in file_path.lower():
            # Most of the questions are with 5 options (A - E)
            # But there are a few minorities
            # 146: (F), 30: (G), 15: (H), 3: (I), 1: (J)
            prediction = extract_answer(completion=response, option_range="a-jA-J")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-jA-J")

        if prediction is not None:
            acc.append(prediction == answer)
            temp = {
                'question': question,
                'output': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)

        else:
            acc.append(False)
            temp = {
                'question': question,
                'output': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)
            invalid_out.append(temp)

    accuracy = sum(acc) / len(acc)
    end_t = time.time()
    elapsed_t = end_t - start_t

    # Print the length of the invalid output and the accuracy
    print('Invalid output length:', len(invalid_out), ', Testing length:', len(acc), ', Accuracy:', accuracy)

    # Save the invalid output in a JSON file
    with open(file_path.replace('.json', '_invalid_responses.json'), 'w') as file:
        json.dump(invalid_out, file, ensure_ascii=False, indent=4)
    # print('Successfully save the invalid output.')

    # Save all the responses in a JSON file
    with open(file_path.replace('.json', '_all_responses.json'), 'w') as file:
        json.dump(all_out, file, ensure_ascii=False, indent=4)
    print(f"Successfully save all the output: Finished performance evaluation in {elapsed_t:.2f} seconds.")





def performance_eval(prompts, answers, args):
    start_t = time.time()
    data_name = args.data_path.split("/")[-1].split("_")[0]
    model_name = args.model_name_or_path.split("/")[-1]
    # 使用 os.path.join() 拼接路径
    file_path = os.path.join(args.save_path, f"{data_name}_{model_name}.json")

    global tokenizer, model
    if model is None or tokenizer is None:
        print(f"\033[32mLoad Checkpoint\033[0m")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     trust_remote_code=True,
                                                     # torch_dtype=torch.float16
                                                     )
        # add special tokens for llama, qwen
        special_tokens_dict = construct_spedical_tokens_dict(tokenizer)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        # model.cuda()

    # Inference
    responses = []
    for i, prompt in enumerate(tqdm(prompts)):
        temp_gen = inference_on_one(prompt, model, tokenizer)
        # post processing
        # temp_gen = temp_gen.split("Answer: ")[-1]
        responses.append(temp_gen)
    print('Successfully finished generating', len(prompts), 'samples!')

    # compute accuracy metric
    acc = []
    all_out = []  # Save all the model's responses
    invalid_out = []  # Only save the invalid responses
    for idx, (question, response, answer) in enumerate(zip(prompts, responses, answers)):
        # question: the medical question
        # response: the model's response
        # answer: the ground truth answer

        # Special Notice:
        # option_range="a-dA-D" means options A, B, C, D
        # option_range="a-eA-E" means options A, B, C, D, E
        # option_range="a-jA-J" means options A, B, C, D, E, F, G, H, I, J
        # English Benchmarks - 11 Benchmarks
        if "pubmedqa" in file_path.lower():
            # We separately write codes to evaluate our models' performance
            # on the PubMedQA benchmark (yes/no/maybe)
            prediction = extract_answer_for_pubmedqa(completion=response)
        elif ("mmlu" in file_path.lower() or "medmcqa" in file_path.lower() or "medqa" in file_path.lower()):
            prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
        elif "english_medexpqa" in file_path.lower():
            prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
        elif "english_usmle" in file_path.lower():
            # Most of the questions are with 5 options (A - E)
            # But there are a few minorities
            # 146: (F), 30: (G), 15: (H), 3: (I), 1: (J)
            prediction = extract_answer(completion=response, option_range="a-jA-J")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-jA-J")

        if prediction is not None:
            acc.append(prediction == answer)
            temp = {
                'question': question,
                'output': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)

        else:
            acc.append(False)
            temp = {
                'question': question,
                'output': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)
            invalid_out.append(temp)

    accuracy = sum(acc) / len(acc)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")

    # Print the length of the invalid output and the accuracy
    print('Invalid output length:', len(invalid_out), ', Testing length:', len(acc), ', Accuracy:', accuracy)

    # Save the invalid output in a JSON file
    with open(file_path.replace('.json', '_invalid_responses.json'), 'w') as file:
        json.dump(invalid_out, file, ensure_ascii=False, indent=4)
    print('Successfully save the invalid output.')

    # Save all the responses in a JSON file
    with open(file_path.replace('.json', '_all_responses.json'), 'w') as file:
        json.dump(all_out, file, ensure_ascii=False, indent=4)
    print('Successfully save all the output.')


def inference_on_one(input_str: Sequence[str], model, tokenizer) -> str:
    # model_inputs = tokenizer(
    #     input_str,
    #     return_tensors='pt',
    #     # padding=True,
    # )
    #
    # topk_output = model.generate(
    #     model_inputs.input_ids.cuda(),
    #     max_new_tokens=1000,
    #     top_k=50
    # )

    input_ids = tokenizer(input_str, padding=True, return_tensors="pt").input_ids
    input_len = input_ids.shape[1]
    input_ids = input_ids.cuda()  # .to(0)
    # 将输入tensor移动到cuda:0设备上
    # inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    generate_ids = model.generate(input_ids, max_new_tokens=64, top_k=50)
    response = tokenizer.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True)[0]

    # output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return "Correct option is " + response
    # return response


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', type=str, default="medqa")
    parser.add_argument('--save_path', type=str,
                        default="/share/project/weiyifan/KG_RAG/kg_rag/prompt_based_generation/MedLLMs/results")
    parser.add_argument('--model_name_or_path', type=str,
                        default="/share/project/weiyifan/KG_RAG/results/checkpoints/output_step2_Llama-3-8B/Llama3-8b-mcts")
    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    # llm = LLM(model=args.model_name_or_path, tensor_parallel_size=2, trust_remote_code=True)  # 使用 vLLM 加载模型
    # args.llm = llm
    print(args)
    medqa_eval(args=args)
    medmcqa_eval(args=args)
    pubmedqa_eval(args=args)

    # mmlu_eval(args=args)
    print("All Medical Datasets have been evaluated !")


if __name__ == "__main__":
    main()
