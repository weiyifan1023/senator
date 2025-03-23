import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from collections import namedtuple
from typing import Dict, List, Tuple, Union
from vllm import LLM, SamplingParams
from tqdm import tqdm
import time, argparse
import pandas as pd
import random

import glob
import jsonlines, json
from benchmark_utils import *
from answer_utils import *

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def load_examples(path: str, seed: int) -> List[Example]:
    """Load questions from csv file and return a list of Example namedtuples."""

    all_data = []
    csv_files = glob.glob(path + "*.csv")
    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)
        all_data.append(temp_df)
    question_df = pd.concat(all_data)
    # question_df = pd.read_csv(path)
    random.seed(seed)

    # filter out rows with Molecular Biology and  Genetics Subdomain
    # 过滤出 Subdomain 为 'Molecular Biology' 或 'Genetics' 的行
    question_df = question_df[
        (question_df['Subdomain'] == 'Genetics')
        # (question_df['Subdomain'] == 'Molecular Biology')
    ]

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        example = Example(row.Question, list_choices[0], list_choices[1], list_choices[2], list_choices[3],
                          list_choices.index(row['Correct Answer']))
        prompt, ans = gpqa_format(example)
        return prompt, ans

    prompts = []
    answers = []
    for _, row in question_df.iterrows():
        prompt, ans = shuffle_choices_and_create_example(row)
        prompts.append(prompt)
        answers.append(ans)

    return prompts, answers





# 4. 推理评测
def evaluate_gpqa(prompts, answers, llm, sampling_params, args):
    start_t = time.time()
    model_name = args.model_name_or_path.split("/")[-1]
    file_path = os.path.join(args.save_path, f"GPQA_{model_name}.json")
    responses = []



    # for i in tqdm(range(0, len(prompts), args.eval_bs)):
    #     batch_prompts = prompts[i:i + args.eval_bs]
    outputs = llm.generate(prompts, sampling_params)  # 完成推理
    for output in outputs:
        tmp_prompt = output.prompt
        if "mcts" in args.model_name_or_path:
            tmp_gen = "Correct option is " + output.outputs[0].text
        else:
            tmp_gen = output.outputs[0].text
        responses.append(tmp_gen)

    print('Successfully finished generating', len(prompts), 'samples!')

    # compute accuracy metric
    acc = []
    all_out = []  # Save all the model's responses
    invalid_out = []  # Only save the invalid responses
    for idx, (question, response, answer) in enumerate(zip(prompts, responses, answers)):

        prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options

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
        json.dump(invalid_out, file, ensure_ascii=False, indent=2)
    print('Successfully save the invalid output.')

    # Save all the responses in a JSON file
    with open(file_path.replace('.json', '_all_responses.json'), 'w') as file:
        json.dump(all_out, file, ensure_ascii=False, indent=2)
    print('Successfully save all the output.')








def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str,
                        default="/share/project/weiyifan/KG_RAG/kg_rag/prompt_based_generation/MedLLMs/results")
    parser.add_argument('--eval_bs', type=int, default=4)
    # /share/project/models/PMC_LLaMA_13B
    # /share/project/models/Llama-3-8B
    # /share/project/models/Qwen2-7B
    # /share/project/weiyifan/KG_RAG/results/checkpoints/output_step2_Llama-3-8B/Llama3-8b-mcts
    parser.add_argument('--model_name_or_path', type=str,
                        default="/share/project/weiyifan/KG_RAG/results/checkpoints/output_step2_Llama-3-8B/Llama3-8b-mcts")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1. 加载 GPQA 数据集
    gpqa_questions, gpqa_answers = load_examples("/share/project/weiyifan/KG_RAG/data/benchmark_data/gpqa/", seed=42)

    # 2. 加载本地模型（使用 vLLM）
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=2, trust_remote_code=True)  # 使用 vLLM 加载模型
    # 3. 定义采样参数（用于生成答案）
    sampling_params = SamplingParams(
        top_k=50,
        max_tokens=64,  # 生成的最大 token 数
    )
    print(args)
    evaluate_gpqa(gpqa_questions, gpqa_answers, llm, sampling_params, args)



if __name__ == "__main__":
    main()
