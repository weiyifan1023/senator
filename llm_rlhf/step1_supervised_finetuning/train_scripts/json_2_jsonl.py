import json
import jsonlines
from datasets import load_dataset

json_file = "/share/project/weiyifan/KG_RAG/data/benchmark_data/PMC_sft_data.json"
output_file = '/share/project/weiyifan/KG_RAG/data/benchmark_data/PMC_sft_data.jsonl'

# json_file = "/share/project/weiyifan/KG_RAG/data/benchmark_data/Huatuo_qa.json"
# output_file = '/share/project/weiyifan/KG_RAG/data/benchmark_data/Huatuo_qa.jsonl'




with open(json_file, 'r') as infile:
    json_data = json.load(infile)

print("Writing JSON data to JSONL file")

with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(json_data)
