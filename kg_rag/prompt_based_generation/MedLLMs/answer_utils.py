# coding=utf-8

import re

# Define translation for Chinese punctuations to English punctuations
chinese_punctuation = "，。；：“”‘’（）【】《》？！—"
english_punctuation = ",.;:\"\"''()[]<>?!-"
translation_table = str.maketrans(chinese_punctuation, english_punctuation)


def extract_answer_for_chinese(completion, option_range="a-eA-E"):
    """
    Extract answer ("A", "B", "C", "D", etc.) from the Chinese or Chinese-English-mixed response
    Special Notice: This answer extraction code for Chinese responses is
    based on the `extract_answer()` function for extracting the answer from English responses
    :param completion: model's response
    :return: the best option to the multi-option problem
    """
    # Please note that we are using two Chinese benchmarks: CMMLU and MCMLE all with 4 options
    if len(completion.strip()) == 1 and completion.strip().upper() in ['A', 'B', 'C', 'D']:
        return completion.strip().upper()

    # Replace some punctuations and space
    completion = completion.replace("*", " ")
    completion = completion.replace(":", "\n\n")
    completion = completion.replace("（", "\n\n")
    completion = completion.replace("）", "\n\n")
    completion = completion.replace("(", "\n\n")
    completion = completion.replace(")", "\n\n")

    # Replace some common Chinese characters to the corresponding English words
    completion = completion.replace("证属", " the correct answer is")
    completion = completion.replace("属于", " the correct answer is")
    completion = completion.replace("宜选", " the correct answer is")

    completion = completion.replace("选项是", " option is")
    completion = completion.replace("选择是", " choice is")
    completion = completion.replace("答案是", " answer is")
    completion = completion.replace("方法是", " answer is")

    completion = completion.replace("可能是", " would be")
    completion = completion.replace("应该是", " would be")  # Not use "should be", as in our codes, we find "would be"
    completion = completion.replace("诊断是", " answer is")  # Not use "diagnosis", as in our codes, we find "answer"

    completion = completion.replace("选项为", " option is")
    completion = completion.replace("选择为", " choice is")
    completion = completion.replace("答案为", " answer is")

    completion = completion.replace("可能为", " would be")
    completion = completion.replace("应该为", " would be")  # Not use "should be", as in our codes, we find "would be"
    completion = completion.replace("诊断为", " answer is")  # Not use "diagnosis", as in our codes, we find "answer"

    completion = completion.replace("最优选项",
                                    " correct answer is")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最好选项",
                                    " correct answer is")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最佳选项",
                                    " correct answer is")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("正确选项", " correct answer is")

    completion = completion.replace("最优的选项",
                                    " correct answer is")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最好的选项",
                                    " correct answer is")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最佳的选项",
                                    " correct answer is")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("正确的选项", " correct answer is")

    completion = completion.replace("是", " is")
    completion = completion.replace("为", " is")
    completion = completion.replace("选", " is")

    completion = completion.replace("最优的", " correct")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最佳的", " correct")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最好的", " correct")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("正确的", " correct")

    completion = completion.replace("最优", " correct")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最佳", " correct")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("最好", " correct")  # Not use "best", as in our codes, we find "correct"
    completion = completion.replace("正确", " correct")

    completion = completion.replace("选项", " option:\n\n")
    completion = completion.replace("选择", " choice:\n\n")
    completion = completion.replace("答案", " answer:\n\n")
    completion = completion.replace("诊断", " answer:\n\n")  # Not use "diagnosis", as in our codes, we find "answer"
    completion = completion.replace("项", " option:\n\n")

    completion = completion.replace("可能", " would ")
    completion = completion.replace("也许", " could ")

    # Apply the translation table to replace all Chinese punctuations with spaces
    completion = completion.translate(translation_table)

    # Replace two spaces to one space
    completion = completion.replace("  ", " ")

    # Add a space to all the Chinese letters
    pattern = re.compile(r'([\u4e00-\u9fff])')
    completion = pattern.sub(r'\1 ', completion)

    return extract_answer(completion=completion, option_range=option_range)


def extract_answer_for_french(completion, option_range="a-eA-E"):
    """
    Extract answer ("A", "B", "C", "D", etc.) from the French-English-mixed responses
    Special Notice: This answer extraction code for French responses is
    based on the `extract_answer()` function for extracting the answer from English responses
    :param completion: model's response
    :return: the best option to the multi-option problem
    """
    # Remove "*" from the sentence
    completion = completion.replace("*", "\n")
    completion = completion.replace("'", "")

    # Remove " la " from the sentence
    completion = completion.replace(" la ", " ")

    # Replace some French words to English
    completion = completion.replace("réponse", "answer")
    completion = completion.replace("correcte", "correct answer")
    completion = completion.replace("La bonne", "correct answer")  # To handle "La bonne réponse est"
    completion = completion.replace("choix", "choice")
    completion = completion.replace("être", "be")
    completion = completion.replace("serait", "would be")
    completion = completion.replace("pourrait", "could")
    completion = completion.replace("est", "is")

    # Change the French special characters to English
    # Special Notice: the reason why we are doing this is that
    # for the following French characters
    # they are not "a-zA-Z" in English,
    # so they are inconsistent with our English answer extraction codes
    # Please check the `additional_patterns` in English answer extraction
    completion = completion.replace("à", "a")
    completion = completion.replace("â", "a")
    completion = completion.replace("À", "A")
    completion = completion.replace("Â", "A")
    completion = completion.replace("ç", "c")
    completion = completion.replace("Ç", "C")
    completion = completion.replace("é", "e")
    completion = completion.replace("è", "e")
    completion = completion.replace("ê", "e")
    completion = completion.replace("ë", "e")
    completion = completion.replace("É", "E")
    completion = completion.replace("È", "E")
    completion = completion.replace("Ê", "E")
    completion = completion.replace("Ë", "E")

    return extract_answer(completion=completion, option_range=option_range)


def extract_answer_for_spanish(completion, option_range="a-eA-E"):
    """
    Extract answer ("A", "B", "C", "D", etc.) from the Spanish-English-mixed responses
    Special Notice: This answer extraction code for Spanish responses is
    based on the `extract_answer()` function for extracting the answer from English responses
    :param completion: model's response
    :return: best option to the multi-option problem
    """
    # Remove "*" from the sentence
    completion = completion.replace("*", "\n")

    # Replace some French words to English
    completion = completion.replace("respuesta", "answer")
    completion = completion.replace("correcta", "correct answer")
    completion = completion.replace("correcto", "correct answer")
    completion = completion.replace("elección", "choice")
    completion = completion.replace("opción", "option")
    completion = completion.replace("haría", "would")
    completion = completion.replace("podría", "could")
    completion = completion.replace("es", "is")
    completion = completion.replace("ser", "be")

    return extract_answer(completion=completion, option_range=option_range)


def extract_answer_for_hindi(completion, option_range="a-eA-E"):
    """
    Extract answer ("A", "B", "C", "D", etc.) from the Hindi-English-mixed responses
    Special Notice: This answer extraction code for Hindi responses is
    based on the `extract_answer()` function for extracting the answer from English responses
    :param completion: model's response
    :return: best option to the multi-option problem
    """
    # Remove "*" from the sentence
    completion = completion.upper().replace("*", "\n")

    # Define the translation dictionary
    translation_dict = {
        'अ': 'A',
        'ब': 'B',
        'स': 'C',
        'द': 'D'
    }

    # Define a function to replace match with corresponding English letter
    def replace_match(match):
        char = match.group(0)
        return translation_dict.get(char, char)

    # Use regular expression to find single letters and replace them
    pattern = r'(?<!\w)([अबसद])(?!\w)'
    completion = re.sub(pattern, lambda m: replace_match(m), completion, count=1)

    return extract_answer(completion=completion, option_range=option_range)


def extract_answer(completion, option_range="a-eA-E"):
    """
    Extract answer ("A", "B", "C", "D", etc.) from the response.
    :param completion: model's response
    :param option_range: range of options to look for (default is 'a-eA-E')
    :return: the best option to the multi-option problem
    """
    # The range of letters to look for (default is '\u4e00-\u9fffa-zA-ZÀ-ÖØ-öø-ÿÁÉÍÓÚÜáéíóúüÑñ\u0900-\u097F')
    potential_letters = "\u4e00-\u9fffa-zA-ZÀ-ÖØ-öø-ÿÁÉÍÓÚÜáéíóúüÑñ\u0900-\u097F"

    # Replace asterisks with newlines for better processing
    completion = completion.replace("*", "\n")

    # Determine the last letter in the option range
    last_letter = option_range[-1]

    # Adjust the option range for patterns excluding 'A'
    if last_letter.lower() == 'b':
        opt_wo_a = "bA-B"
    else:
        opt_wo_a = "b-" + last_letter.lower() + "A-" + last_letter.upper()

    # Extend potential letters to include numbers
    letter_and_num = potential_letters + "0-9"

    # Define patterns to match the correct answer format
    patterns = [
        # Matches "A.", "B.", etc. at the beginning of a line
        re.compile(rf'^([{option_range}])\.'),
        # Matches "correct answer is (A)" and similar formats
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])$'),

        # Matches "correct option is (A)" and similar formats
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])$'),

        # Matches "correct choice is (A)" and similar formats
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])$'),

        # Matches "answer is (A)" and similar formats
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])$'),

        # Matches "option is (A)" and similar formats
        re.compile(rf'[oO]ption is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])$'),

        # Matches "OPTION (A) IS CORRECT" and similar formats
        re.compile(rf'OPTION[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*([{option_range}])[^{letter_and_num}]IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*\(([{option_range}])\)IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*\[([{option_range}])\]IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*\{{([{option_range}])\}}IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*([{option_range}])\)IS CORRECT'),
        re.compile(rf'OPTION[^{potential_letters}]*([{option_range}])$ IS CORRECT'),

        # Matches "choice is (A)" and similar formats
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])$'),

        # Matches "is: (A)" and similar formats
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'),
        # Matches "would be (A)" and similar formats
        re.compile(rf'would be[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'would be[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'would be[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'would be[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(
            rf'would be[^{letter_and_num}]*([{opt_wo_a}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'would be[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'would be[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'would be[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'would be[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'would be[^{potential_letters}]*([{option_range}])$'),

        # Matches "is (A)" and similar formats
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*\(([{option_range}])\)'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*\[([{option_range}])\]'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*\{{([{option_range}])\}}'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'),

        # Matches "be (A)" and similar formats
        re.compile(rf'is[^{letter_and_num}]+([{option_range}])\)'),
        re.compile(rf'be[^{letter_and_num}]+([{option_range}])\)'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*will'),

        # Matches "(A)" followed by any other characters
        re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)$'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]'),
    ]

    # Check for matches with defined patterns
    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return match.group(1).upper()

    # Count occurrences of left and right parentheses
    count_left_paren = completion.count('(')
    count_right_paren = completion.count(')')

    # Adjust patterns if right parentheses are more frequent
    if count_right_paren > count_left_paren:
        patterns = [
            # Matches "(A)" followed by any other characters, adjusted for more right parentheses
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*is'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*would'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*could'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*will'),
            re.compile(rf'^[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)'),
            re.compile(rf':+[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*is'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*would'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*could'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*will'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)$'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)'),
            re.compile(rf'is[^{letter_and_num}]+([{option_range}])\)'),
            re.compile(rf'be[^{letter_and_num}]+([{option_range}])\)'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*is'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*would'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*could'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*will'),
            re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]'),
            re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)$'),
            re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)$'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]'),
        ]

        # Check for matches with the adjusted patterns
        for pattern in patterns:
            match = pattern.search(completion)
            if match:
                return match.group(1).upper()

    # Define additional patterns to match correct answers
    additional_patterns = [
        # Matches "A"
        re.compile(rf"^[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]*$"),

        # Matches "(A) is", "[A] is", "{A} is", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*is'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*is'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*is'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*is'),

        # Matches "(A) would", "[A] would", "{A} would", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*would'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*would'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*would'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*would'),

        # Matches "(A) could", "[A] could", "{A} could", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*could'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*could'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*could'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*could'),

        # Matches "(A) will", "[A] will", "{A} will", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*will'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*will'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*will'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*will'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*will'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*will'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*will'),

        # Matches "option: (A)" and similar formats
        re.compile(rf'[oO]ption:+[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "choice: (A)" and similar formats
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "answer: (A)" and similar formats
        re.compile(rf' is[^{potential_letters}]+\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf' is[^{potential_letters}]+\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf' is[^{potential_letters}]+\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(
            rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]'
        ),
        re.compile(rf' is[^{potential_letters}]+\(([{option_range}])\)$'),
        re.compile(rf' is[^{potential_letters}]+\[([{option_range}])\]$'),
        re.compile(rf' is[^{potential_letters}]+\{{([{option_range}])\}}$'),
        re.compile(rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(rf' is([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf' is([{option_range}])\)$'),
        re.compile(
            rf' is[^{potential_letters}]*[^{letter_and_num}]([{opt_wo_a}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),
        re.compile(rf' is([{opt_wo_a}])[^{letter_and_num}]'),
        re.compile(rf' is([{option_range}])$'),
        re.compile(rf' is[^{potential_letters}]+\(([{option_range}])\)'),
        re.compile(rf' is[^{potential_letters}]+\[([{option_range}])\]'),
        re.compile(rf' is[^{potential_letters}]+\{{([{option_range}])\}}'),
        re.compile(rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)'),
        re.compile(rf' is[^{letter_and_num}]*([{option_range}])\)'),

        # Matches "choice (A)" and similar formats
        re.compile(rf'[cC]hoice[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]hoice[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]hoice[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[cC]hoice[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "answer (A)" and similar formats
        re.compile(rf'[aA]nswer[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[aA]nswer[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[aA]nswer[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[aA]nswer[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[aA]nswer[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[aA]nswer[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[aA]nswer[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "option (A)" and similar formats
        re.compile(rf'[Oo]ption[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[Oo]ption[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[Oo]ption[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[Oo]ption[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[Oo]ption[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),
    ]

    # Check for matches with additional patterns
    for pattern in additional_patterns:
        match = pattern.search(completion)
        if match:
            return match.group(1).upper()

    # Define a pattern to identify multiple options in a sequence
    multi_pattern = re.compile(
        rf'[aA][^{potential_letters}]*[bB][^{potential_letters}]*[cC][^{potential_letters}]*[dD]')
    match_multi = multi_pattern.search(completion)
    if match_multi:
        return None

    # Define sliding window patterns to match correct answers within a short text window
    sliding_window_patterns = [
        re.compile(rf'\(([{option_range}])\)'),
        re.compile(rf'\[([{option_range}])\]'),
        re.compile(rf'\{{([{option_range}])\}}'),
        re.compile(
            rf'[^{letter_and_num}]([{opt_wo_a}])[^{letter_and_num}]'),
    ]

    # Use sliding window approach to find matches within the completion text
    for i in range(len(completion) - 2):
        window = completion[i:i + 3]
        for pattern in sliding_window_patterns:
            match = pattern.search(window)
            if match:
                return match.group(1).upper()

    # If no patterns match, return None
    return None


def helper(str):
    """
    A helper function for PubMedQA Benchmark
    """
    if str == "YES":
        return 'A'
    elif str == "NO":
        return 'B'
    elif str == "MAYBE":
        return "C"
    else:
        return None


def extract_answer_for_pubmedqa(completion):
    """
    Extract the best option from responses for PubMedQA Benchmark
    :param completion: model's response
    :return: the best option to the multi-option problem
    """
    # [aA]nswer is [yY]es/no/maybe
    # [aA]nswer: Yes[^a-zA-Z]
    # ^Yes. ^No^a-zA-Z\s] Maybe.
    # is [yY]es[^a-zA-Z]
    # [^a-zA-Z]Yes.
    # [mM]aybe

    length_1 = len("Yes/No/Maybe")
    length_2 = len("Yes/no")

    pattern_y_n_m = re.compile(r'^([yY][eE][sS])/([nN][oO])/([mM][aA][yY][bB][eE])')
    pattern_y_or_n = re.compile(r'^([yY][eE][sS])/([nN][oO])')

    match_1 = pattern_y_n_m.search(completion)
    match_2 = pattern_y_or_n.search(completion)

    if match_1:
        completion = completion[length_1:]

        patterns = [
            re.compile(r'^[^a-zA-Z]*([yY][eE][sS])'),
            re.compile(r'^[^a-zA-Z]*([nN][oO])'),
            re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])'),
        ]

        # Loop through the patterns and search for a match at the beginning of the string
        for pattern in patterns:
            match = pattern.match(completion)
            if match:
                return helper(match.group(1).upper())

    elif match_2:
        completion = completion[length_2:]

    patterns = [
        re.compile(r'^[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'^[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),
    ]

    # Loop through the patterns and search for a match at the beginning of the string
    for pattern in patterns:
        match = pattern.match(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r'[aA]nswer is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[aA]nswer[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[aA]nswer[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[aA]nswer[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[aA]nswer is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[aA]nswer[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[aA]nswer[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[oO]ption is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[oO]ption[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[oO]ption[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[oO]ption[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[oO]ption is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[oO]ption[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[oO]ption[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[cC]hoice is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[cC]hoice[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[cC]hoice[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[cC]hoice[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[cC]hoice is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[cC]hoice[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[cC]hoice[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),
    ]

    # Loop through the patterns and search for a match
    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r'^[^a-zA-Z]*([yY][eE][sS])$'),
        re.compile(r'^[^a-zA-Z]*([nN][oO])$'),
        re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])$'),
        re.compile(r'^[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),
    ]

    # Loop through the patterns and search for a match at the beginning of the string
    for pattern in patterns:
        match = pattern.match(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r' is[^a-zA-Z]*:+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r' is[^a-zA-Z]*:+[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r' is[^a-zA-Z]*:+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r' is[^a-zA-Z]*:+\s+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*:+\s+[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*:+\s+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r' is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),

        re.compile(r':+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r':+[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r':+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r':+\s+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r':+\s+[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r':+\s+[^a-zA-E]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r' is[^a-zA-Z]*\s+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r' is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r' is[^a-zA-Z]*\s+[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*\s+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[^a-zA-Z]([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),
        re.compile(r'[^a-zA-Z]([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[^a-zA-Z]([yY][eE][sS])$'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])$'),
        re.compile(r'[^a-zA-Z]([nN][oO])$'),

        re.compile(r'[^a-zA-Z]([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])[^a-zA-Z]'),
        re.compile(r'[^a-zA-Z]([nN][oO])[^a-zA-Z]'),

        re.compile(r'[^a-zA-Z]([yY][eE][sS])[^a-zA-Z]*'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])[^a-zA-Z]*'),

        re.compile(r'[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]*'),
        re.compile(r'[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]*'),
    ]

    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r'([nN][oO])[a-zA-Z]+'),
        re.compile(r'[a-zA-Z]+([nN][oO])'),
    ]
    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return None

    extra_pattern_no = re.compile(r'[^a-zA-Z]*([nN][oO])[^a-zA-Z]*')
    match = extra_pattern_no.search(completion)
    if match:
        return helper(match.group(1).upper())

    # If no match is found, return None
    return None


def response_with_option(prompt, response):
    """
    Add the corresponding A/B/C/D option to the prompt
    :param prompt: the prompt to the model
    :param response: the model's response
    :return response: the model's response with the option
    """
    terms = {}
    option_range = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for option in option_range:
        pattern = re.compile(rf'([{option}]\.\s.*?)\n')
        match = pattern.search(prompt)
        if match:
            terms[option] = match.group(1)

    for v in terms.values():
        answer = v[3:].lower()
        response = response.lower()
        response = response.replace(answer, v)

    return response