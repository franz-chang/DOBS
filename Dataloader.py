import torch
import json
import re
import os

# Dataset pick up.
def get_file_path(file_name):
    if file_name == "gsm":
        return './Dataset/GSM-IC_mstep.json'
    elif file_name == "webqa":
        return './Dataset/WebQuestions.json'
    elif file_name == "movqa":
        return './Dataset/MovieQA.json'
    elif file_name == "compx":
        return './Dataset/ComplexWebQuestions.json'
    elif file_name == "SVAMP":
        return './Dataset/SVAMP_reshape.json'
    elif file_name == "strategy":
        return './Dataset/strategyqa_reshape.json'
    elif file_name == "math":
        return './Dataset/math_dataset.json'
    elif file_name == "SQUAD":
        return './Dataset/SQUAD.json'
    elif file_name == "SQUAD2":
        return './Dataset/SQUAD2.json'
    else:
        print("Error!")
        return None 

# Detailed Dataset Loading
def extract_SQUAD(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for item in data['data']:
            for paragraph in item['paragraphs']:
                for qa in paragraph['qas']:
                    if questions_loaded >= q_limits:
                        return qa_dict
                    
                    question = qa.get('question')
                    answers = qa.get('answers', [])
                    if answers:
                        answer = answers[0].get('text')
                        if question and answer:
                            qa_dict[question] = answer
                            questions_loaded += 1

    except json.JSONDecodeError:
        print("Failed to decode JSON from the provided file.")
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    return qa_dict

def extract_SQUAD2(file_path, q_limits=None):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    qa_dict = {}
    question_count = 0
    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                is_impossible = qa['is_impossible']
                # extract Answers
                if not is_impossible:
                    answer = f'"{qa["answers"][0]["text"]}"' if qa['answers'] else "No answer available"
                else:
                    answer = '"No answer available"'  # 处理is_impossible为True的情况，同时使用双引号包围

                # 将问题和答案添加到字典中
                qa_dict[question] = answer
                question_count += 1

                # 如果q_limits被设置了，达到上限就停止
                if q_limits and question_count >= q_limits:
                    return qa_dict

    return qa_dict

# loading datasets functions
def extract_GSM8K(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0  # Initialize a counter to track the number of questions loaded

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each item in the JSON data
        for item in data:
            if q_limits is not None and questions_loaded >= q_limits:
                break  # Stop processing if the question limit is reached

            question = item.get("original_question")
            answer = item.get("answer")
            if question and answer:
                qa_dict[question] = answer
                questions_loaded += 1  # Increment the counter for each loaded question

    except json.JSONDecodeError:
        print("JSON decode error")
    except FileNotFoundError:
        print("file not found")
    except Exception as e:
        print(f"Error：{e}")

    return qa_dict


def extract_movie_qa(file_path, q_limits=None):
    qa_dict = {}
    question_count = 0

    try:
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over all entries in the data
        for entry in data:
            question = entry.get('qText')  # Get the question text
            answers = entry.get('answers', [])  # Get the list of answers, default to empty list if not found

            # Format answers with double quotes and handle multiple answers
            formatted_answer = ', '.join(f'"{ans}"' for ans in answers)

            # Add the question and formatted answers to the dictionary
            qa_dict[question] = formatted_answer
            question_count += 1

            # Stop if the limit is reached, if q_limits is set
            if q_limits and question_count >= q_limits:
                break

    except IOError:
        print(f"Error: File {file_path} cannot be opened.")
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON file.")

    return qa_dict

def extract_compx(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0  # Initialize a counter to track the number of questions loaded

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each item in the JSON data
        for item in data:
            if q_limits is not None and questions_loaded >= q_limits:
                break  # Stop processing if the question limit is reached

            question = item.get("question")
            answers = item.get("answers")
            if answers and len(answers) > 0:
                answer = answers[0].get("answer")
            else:
                answer = None

            if question and answer:
                qa_dict[question] = answer
                questions_loaded += 1  # Increment the counter for each loaded question

    except json.JSONDecodeError:
        print("JSON decode error")
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(f"Error: {e}")

    return qa_dict


def extract_Math(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0  # Initialize a counter to track the number of questions loaded

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each item in the JSON data
        for item in data:
            if q_limits is not None and questions_loaded >= q_limits:
                break  # Stop processing if the question limit is reached

            question = item.get("question")
            answer = item.get("answer")

            if question and answer:
                qa_dict[question] = answer
                questions_loaded += 1  # Increment the counter for each loaded question

    except json.JSONDecodeError:
        print("JSON decode error")
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(f"Error: {e}")

    return qa_dict


def extract_Strategy(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0  # Initialize a counter to track the number of questions loaded

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each item in the JSON data
        for item in data:
            if q_limits is not None and questions_loaded >= q_limits:
                break  # Stop processing if the question limit is reached

            question = item.get("question")
            answer = item.get("answer")

            if question is not None and answer is not None:
                qa_dict[question] = answer
                questions_loaded += 1  # Increment the counter for each loaded question

    except json.JSONDecodeError:
        print("JSON decode error")
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(f"Error: {e}")

    return qa_dict


def extract_SVAMP(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0  # Initialize a counter to track the number of questions loaded

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each item in the JSON data
        for item in data:
            if q_limits is not None and questions_loaded >= q_limits:
                break  # Stop processing if the question limit is reached

            question = item.get("Question")
            answer = item.get("Answer")
            if question and answer:
                qa_dict[question] = answer
                questions_loaded += 1  # Increment the counter for each loaded question

    except json.JSONDecodeError:
        print("JSON decode error")
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(f"Error: {e}")

    return qa_dict

def extract_webqa(file_path, q_limits=100):
    qa_dict = {}
    questions_loaded = 0  # Initialize a counter to track the number of questions loaded

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each item in the JSON data
        for item in data:
            if q_limits is not None and questions_loaded >= q_limits:
                break  # Stop processing if the question limit is reached

            question = item.get("utterance")
            target_value = item.get("targetValue")
            answers = re.findall(r'description \"([^\"]*)\"', target_value)
            answer = answers[0] if answers else None

            if question and answer:
                qa_dict[question] = answer
                questions_loaded += 1  # Increment the counter for each loaded question

    except json.JSONDecodeError:
        print("JSON decode error")
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(f"Error: {e}")

    return qa_dict


def extract_dataset(file_path, file_name, q_limits):
    final_dict_qa = {}
    
    if file_name == "gsm":
        qa_dict = extract_GSM8K(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "webqa":
        qa_dict = extract_webqa(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "movqa":
        qa_dict = extract_movie_qa(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "compx":
        qa_dict = extract_compx(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "SVAMP":
        qa_dict = extract_SVAMP(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "strategy":
        qa_dict = extract_Strategy(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "math":
        qa_dict = extract_Math(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "SQUAD":
        qa_dict = extract_SQUAD(file_path, q_limits)
        final_dict_qa = qa_dict

    elif file_name == "SQUAD2":
        qa_dict = extract_SQUAD2(file_path, q_limits)
        final_dict_qa = qa_dict

    return final_dict_qa














