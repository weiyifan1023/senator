# Knowledge Structural Entropy Guided Agent to Detect and Repair Knowledge Blind Spots in LLMs

<p align="center">
<a href="https://github.com/weiyifan1023/Neeko/blob/main/LICENSE">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/python-3.9+-blue.svg'>
<img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'>
</p>



## Getting Started
```
git clone https://github.com/weiyifan1023/KSE-Nav.git
cd KSE-Nav
conda create -n KSE-Nav
conda activate KSE-Nav
pip install -r requirements.txt
```

### 1. Data preparation

Download the Instruction Tuning dataset from [PMC-LLaMA](https://huggingface.co/datasets/axiong/pmc_llama_instructions).
Place the entire ./data/benchmark_data folder under the root folder. 

Preprocess your datasets to SFT format by running:

```
cd  llm_rlhf/step1_supervised_finetuning/train_scripts
python preprocessing.py
```

### 2.  MCTS for Knowledge Deficiency and Synthetic Data Generation
Here, we initialize subgraph for MCTS,  and exploration maximum entropy path.  (Customize the search depth)

```python
python -m prompt_based_generation/MedLLMs/gen_synthetic_data.py
```

### 3.  Deficiency Knowledge Repair (SFT)
We take [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) as an example, replace some paths in run_qwen.sh, and then execute:
```
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

```

## Contact

weiyifan@buaa.edu.cn 

