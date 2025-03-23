import jsonlines, json
import re


def convert_jsonl_to_desired_format(input_file, output_file):
    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            # My Synthetic Data
            question = data["Question"].replace("<", "").replace(">", "")
            answer = data["Answer"].strip(".").replace("<", "").replace(">", "")
            count += 1
            # PMC SFT Data
            # question = data["instruction"] + "\n" + data["input"]
            # answer = data["output"]

            # Creating the desired content structure
            conversation = {
                'content': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ]
            }

            # Writing to the output file in the required format
            json.dump(conversation, outfile)
            outfile.write('\n')  # New line after each JSON object

            # if count == 90000:
            #     break
    print("Total lines: ", count)

def is_entity(text):
    """
    判断一个字符串是否是实体（而不是句子）。
    """

    # 规则 1：长度不超过 5 个单词
    if len(text.split()) > 8:
        return False

    # 规则 2：不包含标点符号
    if any(punc in text for punc in [",", "\n"]):
        return False

    # 规则 3：如果包含 "xxx is a xxx" 模式，判定为句子
    if re.search(r"\w+ is a \w+", text):
        return False

    return True


def filter_data(input_file, output_file):
    # 打开输入文件
    pass_count = 0
    total_count = 0  # 总样本数量 165821
    with jsonlines.open(input_file, "r") as reader:
        with jsonlines.open(output_file, "w") as writer:
            for item in reader:
                total_count += 1
                if is_entity(item["Answer"]):
                    pass_count += 1
                    writer.write(item)
    # 输出统计结果
    pass_rate = (pass_count / total_count) * 100
    print(f"\nTotal items: {total_count}")
    print(f"Passed items: {pass_count}")
    print(f"Pass rate: {pass_rate:.2f}%")


ori_file = '/share/project/weiyifan/KG_RAG/data/qwen_datasets_10w.jsonl'
filter_file = '/share/project/weiyifan/KG_RAG/data/qwen_datasets_10w_filter.jsonl'
sft_file = '/share/project/weiyifan/KG_RAG/data/qwen_datasets_10w_sft.jsonl'
# data = []
# with open(filter_file, 'r', encoding="utf-8") as f:
#     for line in f:
#         data.append(json.loads(line))

filter_data(ori_file, filter_file)
convert_jsonl_to_desired_format(filter_file, sft_file)
