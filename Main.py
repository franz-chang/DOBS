# attack test
import argparse
import json
from transformers import BertTokenizer, BertForMaskedLM
from Dataloader import extract_dataset, get_file_path
from Function import Adversarial_sample_generate, judgement_function, calculate_accuracy, calculate_asr
from Util import post_to_ollama, log_and_print
import time
import pickle
import nltk
import logging
from nltk import word_tokenize, pos_tag
import warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used")

#################################### Parser Setting ############################################################

# API key Collection
parser = argparse.ArgumentParser(description="Script for processing configuration")
parser.add_argument('--api_key', type=str, default='', help='OpenAI API key')
parser.add_argument('--baidu_key', type=str, default='', help='baidu API key.')
parser.add_argument('--gemini_key', type=str, default='', help="Gemini API key." )

# Inference Settings
parser.add_argument('--question_limits', type=int, default='300', help='The number of loaded questions.')
parser.add_argument('--beam_width', type=int, default= '13000', help='The Beam Width.')
parser.add_argument('--tokenizer', type=str, default='bert-large-uncased', help='The bert model for tokenizer.')
parser.add_argument('--embedding_model', type=str, default='bert-large-uncased', help='The detailed BERT type.')

# Target Model Selection
parser.add_argument('--target_model', type=str, default='phi3.5:3.8b', help="llama3.1:8b, llama3.1:70b, qwen2.5:1.5b, qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, gemma2:2b, gemma2:9b, phi3.5:3.8b")
parser.add_argument('--filename', type=str, default="gsm", help="filename: gsm, webqa, movqa, compx, SVAMP, strategy, math, squad, squad2")

opt = parser.parse_args()
print(opt)


############################################# Main process #########################################################

# File path setting
file_path = get_file_path(opt.filename) 
# Options： opt.webqa_json opt.movqa_json opt.compx_json opt.SVAMP_json opt.strategy_json opt.math_json opt.squaud_json opt.squad2_json

# print("The type:", type(file_path))
qa_dict = extract_dataset(file_path, opt.filename, opt.question_limits)
# filename: gsm, webqa, movqa, compx, SVAMP, strategy, math, squad, squad2

# Score list Initial
score_list_1 = []
score_list_2 = []
time_consumptions = []

# Semantic Variation Dict Stroge.
sem_dict = {}
ls_dict = {}

# Target model selection
model_name = opt.target_model
sigma = 0.8
beam_width = opt.beam_width

# Initial tokenizer and BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


####################################### Clean Attack ####################################
for question, answer in qa_dict.items():
    start_time = time.time()
    response_text = post_to_ollama(model_name, question, opt.filename)
    print("============================================================================================================")
    print(f"Generate Answer: {response_text}")
    print(f"Real Answer:{answer}")
    result = judgement_function(answer, response_text, score_list_1, opt.filename)
    end_time = time.time()
    print("Time Consumption: {:.4f} s".format(end_time - start_time))
    print("============================================================================================================")

####################################### Purtetbuted Attack ####################################
for question, answer in qa_dict.items():
    start_time = time.time()
    Adversarial_sample = Adversarial_sample_generate(question, tokenizer, model, sigma, beam_width, sem_dict, ls_dict)
    response_text = post_to_ollama(model_name, Adversarial_sample, opt.filename)
    print("============================================================================================================")
    print(f"Generate Answer: {response_text}")
    print(f"Real Answer:{answer}")
    result = judgement_function(answer, response_text, score_list_2, opt.filename)
    end_time = time.time()
    time_consumption = end_time - start_time
    time_consumptions.append(time_consumption)
    print("Time Consumption: {:.4f} s".format(time_consumption))
    print("============================================================================================================")
    # Save sem_dict to logs.
    with open('sem_dict.log', 'wb') as file:
        pickle.dump(sem_dict, file)

####################################### Configure logging ####################################
logging.basicConfig(filename='test_results.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s') # time stamp setting

average_time = sum(time_consumptions) / len(time_consumptions)
Clean_acc = calculate_accuracy(score_list_1)

log_and_print("=============================================")
log_and_print("Test Done! Acc is: {}".format(Clean_acc))

Attack_acc = calculate_accuracy(score_list_2)
log_and_print("Attack Done! Acc is: {}".format(Attack_acc))

ASR = calculate_asr(Attack_acc, Clean_acc)
log_and_print("The ASR is: {}".format(ASR))

log_and_print("Test Model: {}".format(opt.target_model))
log_and_print("Test Dataset: {}".format(opt.filename))
log_and_print(f"Average Time Consumption: {average_time:.4f} s")
log_and_print(f"The setting of beam width: {opt.beam_width} ")
log_and_print("=============================================")








