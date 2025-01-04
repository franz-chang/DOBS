import ollama
import logging
from openai import OpenAI


# ollama task distribution
def post_to_ollama(model_name, content, file_name):
    number_files = {"gsm", "SVAMP", "strategy", "math"}
    text_files = {"webqa", "movqa", "compx", "SQUAD", "SQUAD2"}

    if file_name in number_files:
        response_text = post_to_ollama_number(model_name, content)
    elif file_name in text_files:
        response_text = post_to_ollama_text(model_name, content)
    else:
        raise ValueError(f"Unknown file name: {file_name}")

    return response_text


# Situation number responses
def post_to_ollama_text(model_name, content):
    # ICL learning
    basic_content = "Please give me a brief answer directly and promise to answer in English:"
    content = basic_content + content
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': content,
        },
    ])
    response_text = response['message']['content']

    return response_text


# number post API to ollama
def post_to_ollama_number(model_name, content):
    # ICL Learning
    basic_content = "Give me the numerical answers directly, without giving the intermediate steps:"
    content = basic_content + content
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': content,
        },
    ])
    response_text = response['message']['content']

    return response_text


def log_and_print(message):
    logging.info(message)
    print(message)


# OpenAI API post function
def post_to_gpt(model_name, content, file_name):
    number_files = {"gsm", "SVAMP", "strategy", "math"}
    text_files = {"webqa", "movqa", "compx", "SQUAD", "SQUAD2"}

    if file_name in number_files:
        response_text = post_to_gpt_number(model_name, content)
    elif file_name in text_files:
        response_text = post_to_gpt_text(model_name, content)
    else:
        raise ValueError(f"Unknown file name: {file_name}")

    return response_text


# numeric formula post to OpenAI

def post_to_gpt_number(model_name, content):
    client = OpenAI()
    base_content = ("Give me the numerical answers directly in following questions, without giving the intermediate "
                    "steps:")
    final_content = base_content + content
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_content},
        ]
    )

    # Check if there are any choices and if the first choice has a message attribute
    if completion.choices and len(completion.choices) > 0:
        # Properly accessing the message content of the first choice
        response_text = completion.choices[0].message.content  # Corrected access to `content`
    else:
        response_text = "No response generated."

    return response_text


# Text formular post to OpenAI

def post_to_gpt_text(model_name, content):
    client = OpenAI()
    base_content = "Please give me a brief answer directly in following questions and promise to answer in English:"
    final_content = base_content + content
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_content},
        ]
    )

    # Check if there are any choices and if the first choice has a message attribute
    if completion.choices and len(completion.choices) > 0:
        # Properly accessing the message content of the first choice
        # Assuming the API version you're using, 'message' should probably be accessed as an object, not a dictionary
        if hasattr(completion.choices[0], 'message'):
            # Check if message is an object and 'content' can be accessed directly
            response_text = completion.choices[0].message.content
        elif hasattr(completion.choices[0], 'text'):
            response_text = completion.choices[0].text
        else:
            response_text = "No valid response attribute found."
    else:
        response_text = "No response generated."

    return response_text
