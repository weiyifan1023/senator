import json

def convert_jsonl_to_desired_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            # My Synthetic Data
            question = data["Question"]
            answer = data["Answer"].strip(".")

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

# Example usage
# /share/project/weiyifan/KG_RAG/data/benchmark_data/PMC_sft_data.jsonl

# input_file = '/share/project/weiyifan/KG_RAG/data/benchmark_data/PMC_sft_data.jsonl'
# output_file = '/share/project/weiyifan/KG_RAG/data/benchmark_data/PMC_sft_data_llama.jsonl'
input_file = '/share/project/weiyifan/KG_RAG/data/datasejsonlts_10w.'
output_file = '/share/project/weiyifan/KG_RAG/data/datasets_10w_llama.jsonl'
convert_jsonl_to_desired_format(input_file, output_file)
