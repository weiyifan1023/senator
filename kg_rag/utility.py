import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import json
import openai
import os
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from dotenv import load_dotenv, find_dotenv
import torch
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from kg_rag.config_loader import *
import ast
import requests
import math
from itertools import chain
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import cuts
from kg_rag.kg_mcts import *

memory = Memory("cachegpt", verbose=0)

# Config openai library
config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version

torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


# 这里存在问题，需要根据entity类别进行填充。
SPOKE_dict = {
    "AFFECTS_CamG": "In the field of oncology, compound {} affects mutated gene",
    "ASSOCIATES_DaG": "In genetics, disease {} associates with gene",
    "BINDS_CbP": "In biochemistry, compound {} binds protein",
    "BINDS_CbPD": "In biochemistry, compound {} binds with protein domain",
    "CATALYZES_ECcR": "Enzymatic activity {} catalyzes metabolic reaction",
    "CAUSES_CcSE": "Symptom {} can be caused by the side effect of Compound", # "Compound {} causes side effect"
    "CAUSES_OcD": "Organism {} may cause disease",
    "CLEAVESTO_PctP": "Protein {} cleaves into protein",
    "CONSUMES_RcC": "In metabolic reactions, reaction {} consumes compound",
    "CONTAINS_FcC": "Food {} contains compound",
    "CONTRAINDICATES_CcD": "Compound {} is contraindicated in disease",
    "DECREASEDIN_PdD": "Protein {} is decreased in disease",
    "DOWNREGULATES_AdG": "Gene {} downregulated in tissue",
    "DOWNREGULATES_CdG": "Compound {} downregulates gene",
    "DOWNREGULATES_GPdG": "Gene product {} downregulates gene",
    "DOWNREGULATES_KGdG": "Knocked-out gene {} downregulates target gene",
    "DOWNREGULATES_OGdG": "Overexpressed gene {} downregulates target gene",
    "ENCODES_GeP": "Gene {} encodes protein",
    "ENCODES_OeP": "Organism {} encodes protein",
    "EXPRESSES_ACTeG": "Anatomy cell type {} expresses gene",
    "EXPRESSES_AeG": "Anatomy {} expresses gene",
    "HAS_PhEC": "Protein {} has enzymatic activity",
    "INCLUDES_PCiC": "Pharmacological class {} includes compound",
    "INCREASEDIN_PiD": "Protein {} is increased in disease",
    "INTERACTS_PDiPD": "Protein domain {} interacts with protein domain",
    "INTERACTS_PiP": "Protein {} interacts with protein",
    "ISA_AiA": "Anatomy {} is a type of anatomy",
    "ISA_DiD": "Disease {} is a type of disease",
    "ISA_ECiEC": "Enzymatic activity {} is a type of enzymatic activity",
    "ISA_FiF": "Food {} is a type of food",
    "ISA_OiO": "Organism {} is a type of organism",
    "ISA_PWiPW": "Pathway {} is a type of pathway",
    "ISIN_ACTiiA": "Anatomy cell type {} is localized in anatomy",
    "ISIN_ACTiiCT": "Anatomy cell type {} belongs to cell type",
    "LOCALIZES_DlA": "Disease {} localizes to anatomy",
    "MEMBEROF_PDmPF": "Protein domain {} is a member of protein family",
    "PARTICIPATES_GpBP": "Gene {} participates in biological process",
    "PARTICIPATES_GpCC": "Gene {} participates in cellular component",
    "PARTICIPATES_GpMF": "Gene {} participates in molecular function",
    "PARTICIPATES_GpPW": "Gene {} participates in pathway",
    "PARTOF_ApA": "Anatomy {} is part of anatomy",
    "PARTOF_PDpP": "Protein domain {} is part of protein",
    "PARTOF_PpPwG": "Protein {} is part of pathway group",
    "PARTOF_PwpPw": "Pathway group {} is part of pathway group",
    "PRESENTS_DpS": "Disease {} presents symptom",
    "PRESENTS_GpS": "Gene {} associates with symptom",
    "PRODUCES_RpC": "Reaction {} produces compound",
    "RESEMBLES_DrD": "Disease {} resembles disease",
    "TRANSPORTS_PtC": "Protein {} transports compound",
    "TREATS_CtD": "Compound {} treats disease",
    "UPREGULATES_AuG": "Gene {} upregulated in tissue",
    "UPREGULATES_CuG": "Compound {} upregulates gene",
    "UPREGULATES_GPuG": "Gene product {} upregulates gene",
    "UPREGULATES_KGuG": "Knocked-out gene {} upregulates target gene",
    "UPREGULATES_OGuG": "Overexpressed gene {} upregulates target gene"
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed} ...")


def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def get_context_using_spoke_api(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth': config_data['depth']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append(
                        (item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append(
                        (item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append(
                    (item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x: "pubmedId:" + x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:
                        provenance = "SPOKE-KG"
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append(
                (item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    # merge_1.loc[:, "node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name": "source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    # merge_2.loc[:, "node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name": "target"})
    # 创建一个新的列 "prompt"，根据 "edge_type" 中的 label 映射对应的 prompt
    merge_2['prompt'] = merge_2['edge_type'].map(SPOKE_dict)
    # 删除原始的 "edge_type" 列
    merge_2.drop('edge_type', axis=1, inplace=True)
    # 重命名新的 "prompt" 列为 "edge_type"
    merge_2.rename(columns={'prompt': 'edge_type'}, inplace=True)
    #
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x: x.split("_")[0])
    merge_2.loc[:,
    "context"] = merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')
    context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + \
               node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + \
               node_context[0]["data"]["properties"]["source"] + "."
    return context, merge_2


#         if edge_evidence:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + " and attributes associated with this association is in the following JSON format:\n " + merge_2.evidence.astype('str') + "\n\n"
#         else:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + ". "
#         context = merge_2.context.str.cat(sep=' ')
#         context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
#     return context


def get_kg_using_spoke_api(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth': config_data['depth']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append(
                        (item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append(
                        (item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append(
                    (item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x: "pubmedId:" + x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:
                        provenance = "SPOKE-KG"
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append(
                (item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    # merge_1.loc[:, "node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name": "source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    # merge_2.loc[:, "node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name": "target"})
    # 创建一个新的列 "prompt"，根据 "edge_type" 中的 label 映射对应的 prompt
    merge_2['prompt'] = merge_2['edge_type'].map(SPOKE_dict)
    # 删除原始的 "edge_type" 列
    merge_2.drop('edge_type', axis=1, inplace=True)
    # 重命名新的 "prompt" 列为 "edge_type"
    merge_2.rename(columns={'prompt': 'edge_type'}, inplace=True)
    merge_2 = merge_2[["source", "edge_type", "target"]]
    return merge_2


def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + system_prompt + instruction + E_INST
    return prompt_template


class LLMModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': "[PAD]",
                                           "sep_token": "<s>",
                                           "eos_token": "</s>",
                                           "cls_token": "<CLS>",
                                           "mask_token": "<mask>",
                                           "bos_token": "<BOS>",
                                           # "unk_token": "<unk>"
                                           })
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map='auto',
                                                          trust_remote_code=True,
                                                          # torch_dtype=torch.float16
                                                          )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.prompt = """For given facts, generate a question and its corresponding answer. The question should be designed to inquire about the relationship or classification described in the triples, and the answer should be an entity mentioned in the provided facts.
Facts: 
Disease <Thyroid Gland Mucoepidermoid Carcinoma> is a type of disease <thyroid gland carcinoma>.
Compound <Liothyronine> treats disease <thyroid gland carcinoma>.
Question: What compound can be used to treat Thyroid Gland Mucoepidermoid Carcinoma?
Answer: Liothyronine.

Facts:
Disease <thyroid gland carcinoma> resembles disease <ganglioneuroma> 
Disease {ganglioneuroma} presents Symptom <Diarrhea>
Question: What symptom is associated with the disease that resembles thyroid gland carcinoma?
Answer: Diarrhea.

Facts: 
Disease <head and neck cancer> resembles <thyroid gland carcinoma>.
Disease <head and neck cancer> presents Symptom <Dysphonia>.
Disease <head and neck cancer> presents Symptom <Neck Pain>.
Disease <thyroid gland carcinoma> presents Symptom <Dysphonia>.
Disease <thyroid gland carcinoma> presents Symptom <Neck Pain>.
Compound <Paclitaxel> treats disease <head and neck cancer>.
Question: What disease is similar to thyroid gland carcinoma, with Symptom Dysphonia and Neck Pain.
Answer: head and neck cancer.

Facts:
"""

    def gen_synthetic_data(self, facts):
        generate_kwargs = {
            "max_new_tokens": 64,
            "min_new_tokens": 1,
            "temperature": 0.7,  # 设置温度为 0.7 来控制随机性（0 是完全确定，1 是更高随机性）
            "do_sample": True,  # 启用采样
            "top_p": 0.9,  # 核采样，控制累计概率
            "top_k": 50,  # 只考虑概率前50个词
            "penalty_alpha": 0.6,  # 控制生成的多样性
            # "no_repeat_ngram_size": no_repeat_ngram_size,
            # **generation_config,
        }

        input = self.prompt + facts + "\n"
        # feed to LLMs and Generate
        with torch.no_grad():
            input_ids = self.tokenizer(input, return_tensors="pt").input_ids
            input_len = input_ids.shape[1]
            input_ids = input_ids.to(0)
            generate_ids = self.model.generate(input_ids, **generate_kwargs)
            response = self.tokenizer.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0]

        pred_answer = response.strip('\n').split("\n\n")[0]
        return pred_answer

    def gen_synthetic_data_batch(self, facts_batch):
        generate_kwargs = {
            "max_new_tokens": 128,
            "min_new_tokens": 1,
            "temperature": 0.7,  # 设置温度为 0.7 来控制随机性（0 是完全确定，1 是更高随机性）
            "do_sample": True,  # 启用采样
            "top_p": 0.9,  # 核采样，控制累计概率
            "top_k": 50,  # 只考虑概率前50个词
            "penalty_alpha": 0.6,  # 控制生成的多样性
            # "no_repeat_ngram_size": no_repeat_ngram_size,
            # **generation_config,
        }

        # 假设 facts_batch 是一个包含多个文本的列表
        # 对批量输入的文本进行处理
        input_batch = [self.prompt + facts for facts in facts_batch]

        # feed to LLMs and Generate
        with torch.no_grad():
            # 对每个输入进行编码
            input_ids_batch = self.tokenizer(
                input_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).input_ids
            input_len_batch = input_ids_batch.shape[1]
            input_ids_batch = input_ids_batch.cuda()  # 确保将数据转移到正确的设备上（如GPU）

            # 批量生成
            generate_ids_batch = self.model.generate(input_ids_batch, **generate_kwargs)

            # 解码生成的结果，返回每个生成文本
            responses_batch = self.tokenizer.batch_decode(
                generate_ids_batch[:, input_len_batch:], skip_special_tokens=True
            )

        # 处理每个生成结果，并返回
        pred_answers_batch = [response.strip('\n').split("\n\n")[0] for response in responses_batch]

        return pred_answers_batch


def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False,
                method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # (model_name,revision=branch_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                                     torch_dtype=torch.float16)  # ,device_map='auto',torch_dtype=torch.float16,revision=branch_name,cache_dir=cache_dir
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name,
                                                                revision=branch_name,
                                                                cache_dir=cache_dir,
                                                                legacy=False)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                              device_map='auto',
                                                              torch_dtype=torch.float16,
                                                              revision=branch_name,
                                                              cache_dir=cache_dir)
    if not stream:
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens=max_new_tokens,
                        do_sample=True
                        )
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        streamer=streamer
                        )
    llm = HuggingFacePipeline(pipeline=pipe,
                              model_kwargs={"temperature": temperature, "top_p": top_p})
    return llm


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    # print('Calling OpenAI...')
    response = openai.ChatCompletion.create(
        temperature=temperature,
        # deployment_id=chat_deployment_id,
        # model=chat_model_id,
        engine=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    if 'choices' in response \
            and isinstance(response['choices'], list) \
            and len(response) >= 0 \
            and 'message' in response['choices'][0] \
            and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'


@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)


def stream_out(output):
    CHUNK_SIZE = int(round(len(output) / 50))
    SLEEP_TIME = 0.1
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i + CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    print("\n")


def get_gpt35():
    # gpt-3.5-turbo-16k
    # chat_model_id = 'gpt-35-turbo' if openai.api_type == 'azure' else 'gpt-3.5-turbo'
    chat_model_id = 'gpt-4' if openai.api_type == 'azure' else 'gpt-3.5-turbo-16k'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id


def disease_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    resp = get_GPT_response(text, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id,
                            temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None


def disease_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id,
                            chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None


def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)


def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)


def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold,
                     context_sim_min_threshold, edge_evidence, api=True):
    # entities = disease_entity_extractor_v2(question)
    entities = disease_entity_extractor_v2(question)
    node_hits = []
    if entities:
        max_number_of_high_similarity_context_per_node = int(context_volume / len(entities))
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node_name in node_hits:
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context, context_table = get_context_using_spoke_api(node_name)
                return context_table
            node_context_list = node_context.split(". ")
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1),
                                              np.array(node_context_embedding).reshape(1, -1)) for
                            node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if
                                       s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            if edge_evidence:
                high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:,
                "context"] = context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype(
                    'str') + "\n\n"
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted
    else:
        node_hits = vectorstore.similarity_search_with_score(question, k=5)
        max_number_of_high_similarity_context_per_node = int(context_volume / 5)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node in node_hits:
            node_name = node[0].page_content
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context, context_table = get_context_using_spoke_api(node_name)
            node_context_list = node_context.split(". ")
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1),
                                              np.array(node_context_embedding).reshape(1, -1)) for
                            node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if
                                       s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            if edge_evidence:
                high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:,
                "context"] = context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype(
                    'str') + "\n\n"
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted


def interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, llm_type,
                edge_evidence, system_prompt, api=True, llama_method="method-1"):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_v2(question)
    max_number_of_high_similarity_context_per_node = int(config_data["CONTEXT_VOLUME"] / len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")

    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")

    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        if not api:
            node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
        else:
            context, context_table = get_context_using_spoke_api(node_name)
            node_context.append(context)
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    question_embedding = embedding_function_for_context_retrieval.embed_query(question)
    node_context_extracted = ""
    for node_name in node_hits:
        if not api:
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        else:
            node_context, context_table = get_context_using_spoke_api(node_name)
        node_context_list = node_context.split(". ")
        node_context_embeddings = embedding_function_for_context_retrieval.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1),
                                          np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in
                        node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities],
                                             config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > config_data[
            "QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        if edge_evidence:
            high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            context_table.loc[:,
            "context"] = context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype(
                'str') + "\n\n"
            node_context_extracted += context_table.context.str.cat(sep=' ')
        else:
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")

    input("Press enter for Step 5 - LLM prompting")
    print("Prompting ", llm_type)
    if llm_type == "llama":
        from langchain import PromptTemplate, LLMChain
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"],
                          config_data["LLM_CACHE_DIR"], stream=True, method=llama_method)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run(context=node_context_extracted, question=question)
    elif "gpt" in llm_type:
        enriched_prompt = "Context: " + node_context_extracted + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, system_prompt, llm_type, llm_type,
                                  temperature=config_data["LLM_TEMPERATURE"])
        stream_out(output)


def construct_graph(df, model=None, tokenizer=None):
    model_name = "/share/project/models/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # (model_name,revision=branch_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                                 torch_dtype=torch.float16)  # ,device_map='auto',torch_dtype=torch.float16,revision=branch_name,cache_dir=cache_dir

    # 创建一个无向图
    G = nx.MultiGraph()  # nx.Graph()

    # 遍历 DataFrame，添加节点和边到图中
    for index, row in df.iterrows():
        source = row['source']
        target = row['target']
        relation = row['edge_type']  # relation prompt
        prompt = relation.format(source)
        w = get_self_confidence(prompt, target, model, tokenizer)
        # 添加边，不需要显式添加节点，因为 networkx 会自动添加
        G.add_edge(source, target, weight=w)

    # 打印图的基本信息
    print("图的节点数量:", len(G.nodes))
    print("图的边数量:", len(G.edges))

    # 可选：显示图的节点和边
    print("节点:", G.nodes)
    print("边:", G.edges)
    alg_SE = StructEntropy(G)
    print(alg_SE.calc_1dSE())


def get_self_confidence(query, answer, model=None, tokenizer=None):
    """
    This function is used to get the self-confidence P(a|q) of the LLM;
    Previous work use three model signals to represent the model’s confidence:
    1. Min-Prob, 2. Fst-Prob, 3. Prod-Prob.
    We select prod-prod that score the probability of the sum of all tokens.
    """

    # 将问题q和答案a拼接成模型的输入
    input_text = query + " " + answer

    # 对输入进行tokenize
    inputs = tokenizer(input_text, return_tensors="pt")

    # 获取模型的输出logits
    with torch.no_grad():
        outputs = model(**inputs)

    # 计算答案的概率P(a|q)
    # 模型输出的logits是预测每个token的概率
    logits = outputs.logits
    # 获取答案部分的token索引
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    # 计算答案部分的logits对应的log概率
    log_prob = 0
    for i, token_id in enumerate(answer_ids):
        # 对应位置的logits的log概率
        index = len(inputs.input_ids[0]) - len(answer_ids)
        token_log_prob = torch.log_softmax(logits[0, index + i], dim=-1)
        log_prob += token_log_prob[token_id]

    # 将log概率转换为概率
    probability = torch.exp(log_prob).item()
    print(f"P(a|q) = {probability}")
    print(f"-log P(a|q) = {-log_prob.item()}")
    return -log_prob.item()


# q = "President of the United States is"
# a = "Donald John Trump"
# get_self_confidence(q, a)


class StructEntropy:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        self.struc_data_2d = {}  # {comm1: {comm2: [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], comm3: [], ...}, ...}

    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight='weight')

    def calc_1dSE(self):
        '''
        get the 1D SE of the graph
        '''
        SE = 0
        for n in self.graph.nodes:
            d = cuts.volume(self.graph, [n], weight='weight')
            SE += - (d / self.vol) * math.log2(d / self.vol)
        return SE

    def update_1dSE(self, original_1dSE, new_edges):
        '''
        get the updated 1D SE after new edges are inserted into the graph
        '''

        affected_nodes = []
        for edge in new_edges:
            affected_nodes += [edge[0], edge[1]]
        affected_nodes = set(affected_nodes)

        original_vol = self.vol
        original_degree_dict = {node: 0 for node in affected_nodes}
        for node in affected_nodes.intersection(set(self.graph.nodes)):
            original_degree_dict[node] = self.graph.degree(node, weight='weight')

        # insert new edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        self.vol = self.get_vol()
        updated_vol = self.vol
        updated_degree_dict = {}
        for node in affected_nodes:
            updated_degree_dict[node] = self.graph.degree(node, weight='weight')

        updated_1dSE = (original_vol / updated_vol) * (original_1dSE - math.log2(original_vol / updated_vol))
        for node in affected_nodes:
            d_original = original_degree_dict[node]
            d_updated = updated_degree_dict[node]
            if d_original != d_updated:
                if d_original != 0:
                    updated_1dSE += (d_original / updated_vol) * math.log2(d_original / updated_vol)
                updated_1dSE -= (d_updated / updated_vol) * math.log2(d_updated / updated_vol)

        return updated_1dSE

    def calc_node_SE(self, node: str):
        '''
        get the self information I(node) = -log(P(node)) of a node
        '''
        degree = self.graph.degree(node, weight='weight')
        # self_info = - (degree / self.vol) * math.log2(degree / self.vol)
        self_info = - math.log2(degree / self.vol)
        return self_info

    def get_cut(self, comm):
        '''
        get the sum of the degrees of the cut edges of community comm
        '''
        return cuts.cut_size(self.graph, comm, weight='weight')

    def get_volume(self, comm):
        '''
        get the volume of community comm
        '''
        return cuts.volume(self.graph, comm, weight='weight')

    def calc_2dSE(self):
        '''
        get the 2D SE of the graph
        '''
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.degree(node, weight='weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE

    def show_division(self):
        print(self.division)

    def show_struc_data(self):
        print(self.struc_data)

    def show_struc_data_2d(self):
        print(self.struc_data_2d)

    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()

    def update_struc_data(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE of each cummunity,
        then store them into self.struc_data
        '''
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            if volume == 0:
                vSE = 0
            else:
                vSE = - (cut / self.vol) * math.log2(volume / self.vol)
            vnodeSE = 0
            for node in comm:
                d = self.graph.degree(node, weight='weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]

    def update_struc_data_2d(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE after merging each pair of cummunities,
        then store them into self.struc_data_2d
        '''
        self.struc_data_2d = {}  # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
        comm_num = len(self.division)
        for i in range(comm_num):
            for j in range(i + 1, comm_num):
                v1 = list(self.division.keys())[i]
                v2 = list(self.division.keys())[j]
                if v1 < v2:
                    k = (v1, v2)
                else:
                    k = (v2, v1)

                comm_merged = self.division[v1] + self.division[v2]
                gm = self.get_cut(comm_merged)
                vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                    vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                    vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                else:
                    vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                    vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(
                        self.struc_data[v1][0] / vm) + \
                               self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(
                        self.struc_data[v2][0] / vm)
                self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]

    def init_division(self):
        '''
        initialize self.division such that each node assigned to its own community
        '''
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

    def add_isolates(self):
        '''
        add any isolated nodes into graph
        '''
        all_nodes = list(chain(*list(self.division.values())))
        all_nodes.sort()
        edge_nodes = list(self.graph.nodes)
        edge_nodes.sort()
        if all_nodes != edge_nodes:
            for node in set(all_nodes) - set(edge_nodes):
                self.graph.add_node(node)

    def update_division_MinSE(self):
        '''
        greedily update the encoding tree to minimize 2D SE
        '''

        def Mg_operator(v1, v2):
            '''
            MERGE operator. It calculates the delta SE caused by mergeing communities v1 and v2,
            without actually merging them, i.e., the encoding tree won't be changed
            '''
            v1SE = self.struc_data[v1][2]
            v1nodeSE = self.struc_data[v1][3]

            v2SE = self.struc_data[v2][2]
            v2nodeSE = self.struc_data[v2][3]

            if v1 < v2:
                k = (v1, v2)
            else:
                k = (v2, v1)
            vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
            return delta_SE

        # continue merging any two communities that can cause the largest decrease in SE,
        # until the SE can't be further reduced
        while True:
            comm_num = len(self.division)
            delta_SE = 99999
            vm1 = None
            vm2 = None
            for i in range(comm_num):
                for j in range(i + 1, comm_num):
                    v1 = list(self.division.keys())[i]
                    v2 = list(self.division.keys())[j]
                    new_delta_SE = Mg_operator(v1, v2)
                    if new_delta_SE < delta_SE:
                        delta_SE = new_delta_SE
                        vm1 = v1
                        vm2 = v2

            if delta_SE < 0:
                # Merge v2 into v1, and update the encoding tree accordingly
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)

                volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
                cut = self.get_cut(self.division[vm1])
                vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0] / self.vol) * math.log2(
                    self.struc_data[vm1][0] / volume) + \
                           self.struc_data[vm2][3] - (self.struc_data[vm2][0] / self.vol) * math.log2(
                    self.struc_data[vm2][0] / volume)
                self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE]
                self.struc_data.pop(vm2)

                struc_data_2d_new = {}
                for k in self.struc_data_2d.keys():
                    if k[0] == vm2 or k[1] == vm2:
                        continue
                    elif k[0] == vm1 or k[1] == vm1:
                        v1 = k[0]
                        v2 = k[1]
                        comm_merged = self.division[v1] + self.division[v2]
                        gm = self.get_cut(comm_merged)
                        vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                        if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                        else:
                            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(
                                self.struc_data[v1][0] / vm) + \
                                       self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(
                                self.struc_data[v2][0] / vm)
                        struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break


def vanilla_2D_SE_mini(weighted_edges):
    '''
    vanilla (greedy) 2D SE minimization
    '''
    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)

    seg = StructEntropy(g)
    seg.init_division()
    # seg.show_division()
    SE1D = seg.calc_1dSE()

    seg.update_struc_data()
    # seg.show_struc_data()
    seg.update_struc_data_2d()
    # seg.show_struc_data_2d()
    initial_SE2D = seg.calc_2dSE()

    seg.update_division_MinSE()
    communities = seg.division
    minimized_SE2D = seg.calc_2dSE()

    return SE1D, initial_SE2D, minimized_SE2D, communities


def test_vanilla_2D_SE_mini():
    weighted_edges = [(1, 2, 2), (1, 3, 4)]

    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)
    A = nx.adjacency_matrix(g).todense()
    print('adjacency matrix: \n', A)
    print('g.nodes: ', g.nodes)
    print('g.edges: ', g.edges)
    print('degrees of nodes: ', list(g.degree(g.nodes, weight='weight')))

    SE1D, initial_SE2D, minimized_SE2D, communities = vanilla_2D_SE_mini(weighted_edges)
    print('\n1D SE of the graph: ', SE1D)
    print('initial 2D SE of the graph: ', initial_SE2D)
    print('the minimum 2D SE of the graph: ', minimized_SE2D)
    print('communities detected: ', communities)
    return

# data = {
#     'source': [
#         'Bardet-Biedl syndrome 22',
#         'Setmelanotide',
#         'coloboma',
#         'Laurence-Moon syndrome',
#         'hypogonadism',
#         'atrioventricular septal defect',
#         'retinitis pigmentosa',
#         'retinitis pigmentosa'
#     ],
#     'edge_type': [
#         'Disease {} is a type of disease',
#         'Compound {} treats disease',
#         'Disease {} resembles disease',
#         'Disease {} resembles disease',
#         'Disease {} resembles disease',
#         'Disease {} resembles disease',
#         'Disease {} resembles disease',
#         'Compound {} treats disease',
#     ],
#     'target': [
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#         'Bardet-Biedl syndrome',
#
#     ]
# }
# # 转换为 pandas DataFrame
# df = pd.DataFrame(data)
# # construct_graph(df)
# # Load LLM model
# model_name = "/share/project/models/Llama-2-13b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
# # Load knowledge graph and create query interface
# kg = KnowledgeGraph(tokenizer, model)
# kg.load_from_df(df)
# kg_query = KGQuery(kg)
# alg_SE = StructEntropy(kg.graph)
# mcts = MCTS(model, kg_query, alg_SE, num_simulations=100)
# # root_state = {"current_entity": "Bardet-Biedl syndrome"}
# root_state = "Bardet-Biedl syndrome 22"
# _, mcts_path = mcts.search(root_state)
# evidences = []
# for node in mcts_path[1:]:
#     sub = node.parent.state
#     rel, obj = node.action
#     prompt = rel.format(sub)
#     ans = obj
#     case = prompt + " " + ans
#     evidences.append(case)
#
# # Select the best action based on visit counts
# best_action = max(mcts.root.children, key=lambda c: c.visits).action
# print(alg_SE.calc_1dSE())
