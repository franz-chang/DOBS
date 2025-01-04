import re
import torch
import time
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
from Util import post_to_ollama
import warnings
import nltk
from nltk import word_tokenize, pos_tag
warnings.filterwarnings("ignore", category=FutureWarning)

# Beam Search Abnormal Character Filtering
def filter_english_words(tokenizer):
    """ 
    Filters out tokens to include only valid English words, excluding tokens with special characters,
    [unused] tokens, and subword fragments starting with '##'.
    """
    english_vocab = {}
    for word, idx in tokenizer.vocab.items():
        # Ensure that the token contains only English letters, does not contain '##', and is not part of a subword
        if re.match("^[A-Za-z]+$", word) and '##' not in word:
            english_vocab[word] = idx
    print("Filtered Vocabulary Size:", len(english_vocab))  # Debug: Print the size of the filtered vocabulary
    return english_vocab

# Part of speech check:
def pos_detection(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)

    return tagged_tokens

def beam_search_decoder_with_bert(post, top_k, bert_model, tokenizer, start_tokens, english_vocab):
    batch_size, seq_length, vocab_size = post.shape
    start_token_ids = torch.tensor(
        [english_vocab.get(token, tokenizer.vocab['[UNK]']) for token in start_tokens]).unsqueeze(0).repeat(batch_size,1)
    indices = start_token_ids.unsqueeze(1).repeat(1, top_k, 1)
    log_prob = torch.zeros(batch_size, top_k)

    for i in range(len(start_tokens), seq_length):
        log_post = post[:, i, :].log()

        # Apply English word filtering
        mask = torch.full((vocab_size,), float('-inf'))
        for word_id in english_vocab.values():
            mask[word_id] = 0
        log_post += mask

        log_prob = log_prob.unsqueeze(-1) + log_post.unsqueeze(1).repeat(1, top_k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(top_k, largest=True, sorted=True)
        index = index % vocab_size
        index = index.view(batch_size, top_k, 1)
        indices = torch.cat([indices, index], dim=-1)

    return indices

def get_sentence_embedding(tokenized_phrase, model, tokenizer):
    # Convert tokenized input to tensors
    input_ids = torch.tensor([tokenized_phrase]).to(model.device)
    attention_mask = torch.tensor([[1] * len(tokenized_phrase)]).to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Use the last hidden state
    last_hidden_state = outputs.last_hidden_state
    # Use the CLS token (first token) for sentence embedding
    sentence_embedding = last_hidden_state[:, 0, :].squeeze(0)

    return sentence_embedding

def get_sentence_embedding(tokenized_phrase, model, tokenizer):
    # Convert tokenized input to tensors
    input_ids = torch.tensor([tokenized_phrase]).to(model.device)
    attention_mask = torch.tensor([[1] * len(tokenized_phrase)]).to(model.device)
    
    # Get model outputs, ensure to set return_dict=True and output_hidden_states=True if using Hugging Face's transformers
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)

    # Use the last hidden state from the outputs
    last_hidden_state = outputs.hidden_states[-1]
    # Use the CLS token (first token) for sentence embedding
    sentence_embedding = last_hidden_state[:, 0, :].squeeze(0)
    return sentence_embedding

def Adversarial_sample_generate(start_phrase, tokenizer, model, sigma, beamwidth, sem_dict, ls_dict):
    
    # Tokenization
    tokenized_phrase = tokenizer.encode(start_phrase, add_special_tokens=True)

    # Find the position of the first comma or period
    comma_index = tokenized_phrase.index(tokenizer.encode(',', add_special_tokens=False)[0]) if ',' in start_phrase else len(tokenized_phrase)
    period_index = tokenized_phrase.index(tokenizer.encode('.', add_special_tokens=False)[0]) if '.' in start_phrase else len(tokenized_phrase)

    # Determine the minimum index (first punctuation) and start after it
    start_index = min(comma_index, period_index) + 1  # +1 to start after the punctuation

    # POS tagging (assuming pos_detection function is defined elsewhere)
    pos_tags = pos_detection(tokenized_phrase, tokenizer)

    # To store the predicted words for each position
    predicted_sentence_tokens = tokenized_phrase[:]
    original_sentence_embedding = get_sentence_embedding(tokenized_phrase, model, tokenizer)
    print("original_sentence_embedding shape", original_sentence_embedding.shape)
  
    # Used to record the sequence number of the loop, the chosen_index variable is initialized
    
    loop_count = 0
    chosen_index = None
    
    # Iterate over each token position, starting after the first punctuation
    # Target PoS components: 'VB', 'VBZ', 'VBD', 'VBN', 'JJ', 'NNS'
    for mask_position in range(start_index, len(tokenized_phrase)):
        if pos_tags[mask_position][1] in {'VB', 'VBZ', 'VBD', 'VBN', 'JJ', 'NNS'}:
            masked_tokens = tokenized_phrase[:]
            masked_tokens[mask_position] = tokenizer.mask_token_id

            # Predict masked token using the model, ensure to set return_dict=True and output_hidden_states=True
            with torch.no_grad():
                outputs = model(torch.tensor([masked_tokens]).to(model.device), attention_mask=torch.tensor([[1] * len(masked_tokens)]).to(model.device), return_dict=True, output_hidden_states=True)

            # Extract logits for the masked position
            logits = outputs.logits[0, mask_position]
            softmax = torch.nn.Softmax(dim=-1)
            probs = softmax(logits)

            # Determine the middle index only if chosen_index is None
            if chosen_index is None:
                middle_index = beamwidth // 2
                # middle_index = probs.shape[0] // 2
                chosen_index = middle_index

            predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
            predicted_sentence_tokens[mask_position] = predicted_token_id

            # Calculate cosine similarity
            modified_sentence_embedding = get_sentence_embedding(predicted_sentence_tokens, model, tokenizer)
            cos_sim = F.cosine_similarity(original_sentence_embedding, modified_sentence_embedding, dim=0)
            print("Cosine similarity:", cos_sim.item())

            # Adjust token choice based on cosine similarity
            if cos_sim < sigma and chosen_index < probs.shape[0] - 1:
                # Step X is setting to 50
                chosen_index += 50  # the moving step
                predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
                predicted_sentence_tokens[mask_position] = predicted_token_id
                print("Up moving!")
                print("Index:", chosen_index)
                
            elif cos_sim > sigma and chosen_index > 0:
                new_index = chosen_index - 50  # the moving step
                print("Down Moving!")
                print("Index:", chosen_index)
                # Prevent moving to the last position (ensure it's at least greater than 0)
                if new_index > 0:
                    chosen_index = new_index
                    predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
                    predicted_sentence_tokens[mask_position] = predicted_token_id

            # Store chosen_index in sem_dict, using the sequence number as the key
            sem_dict[loop_count] = chosen_index
            round_cos_sim = round(cos_sim.item(), 3)
            ls_dict[loop_count] = round_cos_sim
            loop_count += 1

    # Decode the fully predicted sentence
    final_predicted_sentence = tokenizer.decode(predicted_sentence_tokens)
    print("Final predicted sentence:", final_predicted_sentence)

    return final_predicted_sentence

    
def indices_to_text(indices, tokenizer):
    sentences = []
    for batch in indices:
        beams = []
        for beam in batch:
            if isinstance(beam, torch.Tensor):
                sentence = tokenizer.decode(beam, skip_special_tokens=True)
            else:
                sentence = tokenizer.decode(beam, skip_special_tokens=True)
            beams.append(sentence)
        sentences.append(beams)
    return sentences

# Mask words check.
def masked_words_check(start_phrase):
    # Initialize tokenizer and filter vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)
    english_vocab = filter_english_words(tokenizer)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)

    # Tokenization
    tokenized_phrase = tokenizer.encode(start_phrase, add_special_tokens=False)

    # Find the position to start the replacements (after the first occurrence of a period)
    period_index = tokenized_phrase.index(
        tokenizer.encode('.', add_special_tokens=False)[0]) + 1  # +1 to start after the period

    # Iterate over each token position, starting after the first period
    for mask_position in range(period_index, len(tokenized_phrase)):
        # Create a copy of the tokens and mask one token
        masked_tokens = tokenized_phrase[:]
        masked_tokens[mask_position] = tokenizer.mask_token_id  # Apply mask

        # Convert to tensors
        input_ids = torch.tensor([masked_tokens])
        attention_mask = torch.tensor([[1] * len(masked_tokens)])

        # Predict masked token using the model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Extract logits for the masked position
        logits = outputs.logits[0, mask_position]
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)

        # Find the most likely token ID
        predicted_token_id = probs.argmax().item()
        predicted_word = tokenizer.decode([predicted_token_id])

        # Replace the masked token with the predicted word in the original token list
        reconstructed_tokens = tokenized_phrase[:]
        reconstructed_tokens[mask_position] = predicted_token_id  # Replace with predicted token ID
        reconstructed_sentence = tokenizer.decode(reconstructed_tokens)

        # print(f"Masked position {mask_position}, prediction: {predicted_word}")
        # print(f"Reconstructed sentence: {reconstructed_sentence}\n")

        return 1

def pos_detection(sentence, tokenizer):
    # Decode tokens to string
    words = [tokenizer.decode([token]) for token in sentence]
    # Join words into a single string (sentence)
    sentence_str = ' '.join(words)
    # Perform POS tagging
    tokens = word_tokenize(sentence_str)
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens



def extract_last_number(string):
    numbers = re.findall(r'\d+', string)
    
    return numbers[-1] if numbers else None

# Judgement Task Distribution

def judgement_function(answers, respones_text, score_list, file_name):
    number_files = {"gsm", "SVAMP", "strategy", "math"}
    text_files = {"webqa", "movqa", "compx", "SQUAD", "SQUAD2"}

    if file_name in number_files:
        result = judgement_function_number(answers, respones_text, score_list)
    elif file_name in text_files:
        result = judgement_function_text(answers, respones_text, score_list)
    else:
        raise ValueError(f"Unknown file name: {file_name}")
    
    return result

def judgement_function_number(answers, respones_text, score_list):
    numbers = re.findall(r'\d+', respones_text)
    if numbers:
        last_number = numbers[-1]
    else:
        last_number = 0.0  
    if float(answers) == float(last_number):
        # print("Correct!")
        result = 1
    else:
        # print("Incorrect!")
        result = -1
    
    score_list.append(result) 
    return result


def clean_and_tokenize(text):
    """
    Cleans the input text by removing all non-alphabet characters and converts to lowercase.
    Then tokenizes the text into a set of words.
    Parameters:
    - text (str): The text to be cleaned and tokenized.
    Returns:
    - set: A set of words from the cleaned text.
    """
    # Remove all non-alphabet characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Split the cleaned text into words and return as a set
    return set(cleaned_text.split())

def judgement_function_text(answers, responses_text, score_list):
    """
    Determines if the intersection of words in two cleansed texts meets the criteria based on the length of answers.
    
    Parameters:
    - answers (str): A string containing the reference answers.
    - responses_text (str): A string containing the response text to evaluate.
    - score_list (list): List to append the score.
    
    Returns:
    - int: Result value of 1 if the criteria are met, otherwise -1.
    """
    # Clean and tokenize both inputs
    answer_words = clean_and_tokenize(answers)
    response_words = clean_and_tokenize(responses_text)
    # Find the intersection of both sets
    common_words = answer_words.intersection(response_words)
    
    # Determine the result based on the number of common words and the number of words in answers
    if (len(answer_words) == 1 and len(common_words) >= 1) or (len(common_words) >= 2):
        # print("Correct!")
        result = 1
    else:
        # print("Incorrect!")
        result = -1

    score_list.append(result) 
    return result

# 正确率计算
def calculate_accuracy(score_list):
    if not score_list:  
        return 0.00  
    total_count = len(score_list)
    correct_count = score_list.count(1)
    accuracy = (correct_count / total_count) * 100  
    result = round(accuracy, 2) 
    # print("Accuracy is:", result)
    
    return result  

def calculate_asr(attack_acc, clean_acc):
    if clean_acc == 0.0:
        final_asr = 0.0
    if attack_acc > 1 or clean_acc > 1:
        attack_acc /= 100
        clean_acc /= 100
        asr = (1 - attack_acc / clean_acc) * 100
        final_asr = round(asr, 2) 
    return final_asr  


