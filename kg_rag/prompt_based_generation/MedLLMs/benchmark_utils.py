# coding=utf-8

import re

# Pre-defined prompts references
# Leaderboard: https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard
# Language Model Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
# Special Notice 1: We use `"Directly answer the best option:"` instead of `Answer:`
# to better guide LLMs to generate the best option and to easier extract the best option from the responses
# Special Notice 2: Since a lot of LLaMA responses were in `mixed English-Hindi`,
# we used "कृपया प्रश्न का उत्तर हिंदी में दें और सीधे सबसे अच्छे विकल्प के साथ जवाब दें:" to guide the model to
# respond the question in Hindi only for LLaMA models
english_prompt = "Directly answer the best option:"
english_prompt_pubmedqa = "Directly answer yes/no/maybe:"
hindi_prompt = "सीधे सबसे अच्छे विकल्प के साथ जवाब दें:"
# For LLaMA models, please use the following prompt for Hindi Benchmarks
# Please answer the question in Hindi and directly answer the best option:
# hindi_prompt = "कृपया प्रश्न का उत्तर हिंदी में दें और सीधे सबसे अच्छे विकल्प के साथ जवाब दें:"
french_prompt = "Répondez directement avec la meilleure option:"
spanish_prompt = "Responde directamente con la mejor opción:"
chinese_prompt = "直接回答最优选项:"

one_shot_prompt = """Directly answer the best option in the following format: "Correct option is [option]. [answer]."
During periradicular surgery bone is removed with?
A. Double inverted cone burr 
B. Round burr
C. Fissure burr 
D. Taper Burr
Correct option is B. Round burr

"""

def gpqa_format(item):
    """
    The format the input of gpqa Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    number_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    english_prompt = "Directly answer the best option:"
    # Start with the question part
    question = item.question + "\n"
    answer = number_to_letter[item.correct_index]

    # Initialize an empty string for the options
    options_str = "A. " + item.choice1 + "\n"
    options_str += "B. " + item.choice2 + "\n"
    options_str += "C. " + item.choice3 + "\n"
    options_str += "D. " + item.choice4 + "\n"

    # Concatenate the question with the formatted options string
    # "Correct best option is"
    full_question = english_prompt + "\n" + question + options_str + "Correct option is"
    return full_question, answer

def medqa_format(item):
    """
    The format the input of MedQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"

    # Initialize an empty string for the options
    options_str = ""

    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["options"], dict) and item["options"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["options"].keys())

        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["options"][key]
            # Append the key and value to the options string with a newline for all except the last one
            options_str += f"{key}. {value}\n"

    # Concatenate the question with the formatted options string
    # full_question = question + options_str + english_prompt
    full_question = english_prompt + "\n" + question + options_str + "Correct option is"
    # full_question = one_shot_prompt + question + options_str + "Correct option is"

    return full_question


def pubmedqa_format(item):
    """
    The format the input of PubMedQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = ("Abstract: " + "\n".join(item["CONTEXTS"])
                + "\nQuestion: " + item["QUESTION"] + "\n")

    # Concatenate the question with the formatted options string
    full_question = question + english_prompt_pubmedqa

    return full_question


def medmcqa_format(item):
    """
    The format the input of MedMCQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"

    # Initialize an empty string for the options
    options_str = "A. " + item["opa"] + "\n"
    options_str += "B. " + item["opb"] + "\n"
    options_str += "C. " + item["opc"] + "\n"
    options_str += "D. " + item["opd"] + "\n"

    # Concatenate the question with the formatted options string
    # full_question = question + options_str + english_prompt
    full_question = english_prompt + "\n" + question + options_str + "Correct option is"
    # full_question = one_shot_prompt + question + options_str + "Correct option is"
    return full_question


def usmle_format(item):
    """
    The format the input of the Internal USMLE QA (Lindsey, Divya, and Meagan)
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"]
    options_str = item["choices"]

    for letter in "ABCDEFGHIJ":
        options_str = re.sub(rf"\({letter}\)", rf"\n{letter}.", options_str)

    # Add a "\n" after all the options
    options_str += "\n"

    # Concatenate the question with the formatted options string
    full_question = question + options_str + english_prompt

    return full_question


def medexpqa_format(item, lang='English'):
    """
    The format the input of MedExpQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["full_question"] + "\n"

    # Initialize an empty string for the options
    options_str = ""

    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["options"], dict) and item["options"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["options"].keys())

        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["options"][key]

            if key == "1":
                key = "A"
            elif key == "2":
                key = "B"
            elif key == "3":
                key = "C"
            elif key == "4":
                key = "D"
            elif key == "5":
                key = "E"

            if value != "None":
                # Append the key and value to the options string with a newline for all except the last one
                options_str += f"{key}. {value}\n"

    # Concatenate the question with the formatted options string
    full_question = question + options_str

    if lang == 'English':
        full_question += english_prompt
    elif lang == 'Spanish':
        full_question += spanish_prompt
    elif lang == 'French':
        full_question += french_prompt

    return full_question


def mmlu_format(item, lang='English'):
    """
    The format the input of the MMLU Benchmark
    (1) anatomy_test.csv
    (2) clinical_knowledge_test.csv
    (3) college_biology_test.csv
    (4) college_medicine_test.csv
    (5) medical_genetics_test.csv
    (6) professional_medicine_test.csv
    :param item: the input one example
    :param lang: the language of the prompt
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item[0] + "\n"

    # Initialize an empty string for the options
    options_str = "A. " + item[1] + "\n"
    options_str += "B. " + item[2] + "\n"
    options_str += "C. " + item[3] + "\n"
    options_str += "D. " + item[4] + "\n"

    # Concatenate the question with the formatted options string
    full_question = english_prompt + "\n" + question + options_str + "Correct option is"

    return full_question


def mcmle_format(item):
    """
    The format the input of MedQA-MCMLE Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"

    # Initialize an empty string for the options
    options_str = ""

    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["options"], dict) and item["options"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["options"].keys())

        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["options"][key]
            # Append the key and value to the options string with a newline for all except the last one
            options_str += f"{key}. {value}\n"

    # Concatenate the question with the formatted options string
    full_question = question + options_str + chinese_prompt

    return full_question


def cmmlu_format(item):
    """
    The format the input of the CMMLU Benchmark
    (1) anatomy.csv
    (2) clinical_knowledge.csv
    (3) college_medicine.csv
    (4) genetics.csv
    (5) nutrition.csv
    (6) traditional_chinese_medicine.csv
    (7) virology.csv
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item[1] + "\n"

    # Initialize an empty string for the options
    options_str = "A. " + item[2] + "\n"
    options_str += "B. " + item[3] + "\n"
    options_str += "C. " + item[4] + "\n"
    options_str += "D. " + item[5] + "\n"

    # Concatenate the question with the formatted options string
    full_question = question + options_str + chinese_prompt

    return full_question


def frenchmedmcqa_format(item):
    """
    The format the input of the FrenchMedMCQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"

    # Initialize an empty string for the options
    options_str = ""

    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["answers"], dict) and item["answers"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["answers"].keys())

        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["answers"][key]
            # Append the key and value to the options string with a newline for all except the last one
            options_str += f"{key.upper()}. {value}\n"

    # Concatenate the question with the formatted options string
    full_question = question + options_str + french_prompt

    return full_question


def headqa_format(item):
    """
    The format the input of the HEAD-QA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["qtext"] + "\n"

    # Initialize an empty string for the options
    options_str = "A. " + item['answers'][0]["atext"] + "\n"
    options_str += "B. " + item['answers'][1]["atext"] + "\n"
    options_str += "C. " + item['answers'][2]["atext"] + "\n"
    options_str += "D. " + item['answers'][3]["atext"] + "\n"

    # Concatenate the question with the formatted options string
    full_question = question + options_str + spanish_prompt

    return full_question



