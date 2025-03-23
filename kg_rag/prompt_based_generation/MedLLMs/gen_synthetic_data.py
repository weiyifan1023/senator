import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

import argparse
import pandas as pd
from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=bool, default=False, help='Flag for interactive mode')
parser.add_argument('-m', type=str, default='method-1', help='Method to choose for Llama model')
parser.add_argument('-e', type=bool, default=True, help='Flag for showing evidence of association from the graph')
# parser.add_argument('-', type=bool, default=True, help='')
args = parser.parse_args()

INTERACTIVE = args.i
METHOD = args.m
EDGE_EVIDENCE = bool(args.e)

SYSTEM_PROMPT = system_prompts["KG_RAG_BASED_TEXT_GENERATION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(
    config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]

INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)


def init_seed_spoke_kg(seed_ent_pool):
    context_list = []

    save_dir = '/share/project/weiyifan/KG_RAG/data/kg_data'
    os.makedirs(save_dir, exist_ok=True)

    for idx, ent in enumerate(seed_ent_pool):
        print("Retrieving context from SPOKE graph... ", idx)
        try:
            context_table = get_kg_using_spoke_api(ent)
        except:
            print("Error occurred while retrieving context for entity: ", ent)
            continue
        file_path = os.path.join(save_dir, f'{ent}.csv')
        context_table.to_csv(file_path, index=False)
        # context_list.append(context_table)
    print("SPOKE KG retrieval complete")


def save_sample_to_json(sample, evidences, value, file_path="/share/project/weiyifan/KG_RAG/data/datasets_1w.jsonl"):
    # 假设sample是字符串，并且格式固定为 "Question: ... Answer: ..."
    try:
        # 提取问题和答案
        question_start = sample.find("Question: ") + len("Question: ")
        answer_start = sample.find("Answer: ") + len("Answer: ")

        # 获取问题和答案内容
        question = sample[question_start:sample.find("Answer: ")].strip()
        answer = sample[answer_start:].strip()

        # 构造字典
        sample_data = {
            "Question": question,
            "Answer": answer,
            "Evidence": ",".join(evidences),
            "Reward": round(value, 2),
        }

        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False)
            f.write("\n")  # 每个 JSON 对象后面加上换行符

        # print(f"Sample saved: {sample_data}")

    except Exception as e:
        print(f"Error parsing sample: {e}")


def main():
    set_seed(42)
    # 初始化KG：使用所有seed entity 构建一个总的KG; 之后对每一个seed entity使用MCTS进行搜索,用于合成数据
    file_path = "/share/project/weiyifan/KG_RAG/data/disease_name_with_id.csv"
    df = pd.read_csv(file_path)
    head_entities = set()
    if 'disease_name' in df.columns:
        head_entities.update(df['disease_name'].dropna())  # dropna() 用于去掉空值

    seed_ent_pool = list(head_entities)
    context_list = []

    cache = True
    if cache:
        kg_data = pd.read_csv('/share/project/weiyifan/KG_RAG/data/kg.csv', dtype={'weight': float})
    else:
        save_dir = '/share/project/weiyifan/KG_RAG/data/kg_data'

        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            context_table = pd.read_csv(file_path)
            context_list.append(context_table)
        final_context = pd.concat(context_list, axis=0, ignore_index=True)
        # 去重
        kg_data = final_context.drop_duplicates()
        kg_data = kg_data[pd.notna(kg_data['edge_type'])]

    # Load LLM model
    model_name = "/share/project/models/Llama-2-13b-chat-hf"
    llm = LLMModel(model_name)
    tokenizer = llm.tokenizer
    model = llm.model

    # Load knowledge graph and create query interface
    kg = KnowledgeGraph(tokenizer, model)

    kg.load_from_df(kg_data)  # .head(200)
    kg_query = KGQuery(kg)
    alg_SE = StructEntropy(kg.graph)
    mcts = MCTS(model, kg_query, alg_SE, num_simulations=100)
    for seed_ent in tqdm(seed_ent_pool):
        try:
            _, mcts_path = mcts.search(seed_ent)
        except:
            print("Entity Node not in KG")
            continue

        evidence_batch = []
        value_batch = []
        for path in mcts_path:
            evidences = []
            path_value = 0
            for node in path[1:]:
                node_value = node.value / (node.visits + 1)
                path_value += node_value
                sub = node.parent.state
                rel, obj = node.action
                ans = obj
                sub = "<" + sub + ">"
                obj = "<" + obj + ">"
                prompt = rel.format(sub)
                case = prompt + " " + obj
                evidences.append(case)

            evidences_str = "\n".join(evidences)
            # single
            sample = llm.gen_synthetic_data(evidences_str)
            save_sample_to_json(sample, evidences, path_value)



    print("Synthetic Dataset Generated Done!")


if __name__ == "__main__":
    main()
