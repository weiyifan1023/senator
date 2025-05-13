# Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs

<p align="center">
<a href="https://github.com/weiyifan1023/Neeko/blob/main/LICENSE">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/python-3.9+-blue.svg'>
<img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'>
</p>

<p align="center">
🔔 <a href="https://github.com/weiyifan1023/senator" target="_blank">Code</a> • 📃 <a href="https://arxiv.org/abs/2505.07184" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/datasets/axiong/pmc_llama_instructions" target="_blank">Dataset</a> <br>
</p>

## Abstract

Large language models (LLMs) have achieved unprecedented performance by leveraging vast pretraining corpora—the "fossil fuel" of modern AI—as predicted by scaling laws. 
However, the diminishing supply of high-quality, human-annotated data, especially in specialized domains, demands a shift toward synthetic data as a new energy source for further advancements. 
In this paper, we propose  a novel Structural Entropy-guided Knowledge Navigator (SENATOR) framework that addresses the intrinsic knowledge deficiencies of LLMs. 
Our approach employs the Structure Entropy (SE) metric to quantify uncertainty along knowledge graph paths and leverages Monte Carlo Tree Search (MCTS) to selectively explore regions where the model lacks domain-specific knowledge. 
Guided by these insights, the framework generates targeted synthetic data for supervised fine-tuning, enabling continuous self-improvement. 
Experimental results on medical benchmarks demonstrate that our SENATOR agent effectively supplements the pretraining corpus by injecting missing domain-specific information, leading to significant performance gains in models such as Llama-3 and Qwen2. 
Our findings highlight the potential of synthetic data as the “new energy” for LLMs, paving the way for more efficient and scalable strategies to sustain and enhance model performance.


## Distribution Analysis

![Image text](https://github.com/weiyifan1023/senator/blob/main/figures/ditribution_contour_group.png)

## Prepare Environments

```
git clone https://github.com/weiyifan1023/senator.git

cd senator
conda create -n senator
conda activate senator
pip install -r requirements.txt
```

## Quick Start

### 1. Data preparation

The seed entities of SPOKE KG are derived from the Project [KG_RAG](https://github.com/BaranziniLab/KG_RAG/tree/main/data)

Download the Instruction Tuning dataset from the Paper [PMC-LLaMA](https://huggingface.co/datasets/axiong/pmc_llama_instructions).

Place the entire ./data/benchmark_data folder under the root folder. 

Preprocess your datasets to SFT format by running:

```python
cd  llm_rlhf/step1_supervised_finetuning/train_scripts
python preprocessing.py
```

### 2.  MCTS for Knowledge Deficiency and Synthetic Data Generation
Here, we initialize subgraph for MCTS,  and exploration maximum entropy path.  

(Customize the search depth)

```python
python -m prompt_based_generation/MedLLMs/gen_synthetic_data.py
```

### 3.  Deficiency Knowledge Repair (SFT)
We take [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) as an example, replace some paths in run_qwen.sh, and then execute:
```sh
cd llm_rlhf/step1_supervised_finetuning
bash train_scripts/qwen2/run_qwen.sh
```

### 4.  Evaluation 

To eval  MedQA, MedMCQA and PubMedQA datasets, you can run:

```python
 cd prompt_based_generation/MedLLMs
 python eval_medical_qa.py
```



## Citation
If you find our paper inspiring and have utilized it in your work, please cite our paper.
```
@misc{wei2025structuralentropyguidedagent,
      title={Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs}, 
      author={Yifan Wei and Xiaoyan Yu and Tengfei Pan and Angsheng Li and Li Du},
      year={2025},
      eprint={2505.07184},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.07184}, 
}
```

## Contact

weiyifan@buaa.edu.cn and duli@baai.ac.cn

