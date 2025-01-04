# Base function Storge

def Adversarial_sample_generate(start_phrase, tokenizer, model, sigma):
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

    # Iterate over each token position, starting after the first punctuation
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

            # Determine the middle index
            middle_index = probs.shape[0] // 2
            chosen_index = middle_index
            predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
            predicted_sentence_tokens[mask_position] = predicted_token_id

            # Calculate cosine similarity
            modified_sentence_embedding = get_sentence_embedding(predicted_sentence_tokens, model, tokenizer)
            cos_sim = F.cosine_similarity(original_sentence_embedding, modified_sentence_embedding, dim=0)
            print("Cosine similarity:", cos_sim.item())

            # Adjust token choice based on cosine similarity
            if cos_sim < sigma and chosen_index < probs.shape[0] - 1:
                chosen_index += 1
                predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
                predicted_sentence_tokens[mask_position] = predicted_token_id
                print("Up moving!")
                print("Index:", chosen_index)
            if cos_sim > sigma and chosen_index > 0:
                new_index = chosen_index - 1
                print("Down Moving!")
                print("Index:", chosen_index)
                # Prevent moving to the last position (ensure it's at least greater than 0)
                if new_index > 0:
                    chosen_index = new_index
                    predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
                    predicted_sentence_tokens[mask_position] = predicted_token_id
    # Decode the fully predicted sentence
    final_predicted_sentence = tokenizer.decode(predicted_sentence_tokens)
    print("Final predicted sentence:", final_predicted_sentence)

    return final_predicted_sentence
    
def Adversarial_sample_generate_topk(start_phrase, tokenizer, model):
    # Tokenization
    tokenized_phrase = tokenizer.encode(start_phrase, add_special_tokens=False)

    # Find the position of the first comma or period
    comma_index = tokenized_phrase.index(
        tokenizer.encode(',', add_special_tokens=False)[0]) if ',' in start_phrase else len(tokenized_phrase)
    period_index = tokenized_phrase.index(
        tokenizer.encode('.', add_special_tokens=False)[0]) if '.' in start_phrase else len(tokenized_phrase)

    # Determine the minimum index (first punctuation) and start after it
    start_index = min(comma_index, period_index) + 1  # +1 to start after the punctuation

    # POS tagging
    pos_tags = pos_detection(tokenized_phrase, tokenizer)

    # To store the predicted words for each position
    predicted_sentence_tokens = tokenized_phrase[:]

    # Iterate over each token position, starting after the first punctuation
    for mask_position in range(start_index, len(tokenized_phrase)):
        # Check if the current word's POS is in the targeted POS list
        if pos_tags[mask_position][1] in {'VB', 'VBZ', 'VBD', 'VBN', 'JJ', 'NNS'}:
            # Create a copy of the tokens and mask one token
            masked_tokens = tokenized_phrase[:]
            masked_tokens[mask_position] = tokenizer.mask_token_id  # Apply mask

            # Convert to tensors
            input_ids = torch.tensor([masked_tokens])
            attention_mask = torch.tensor([[1] * len(masked_tokens)])

            # Predict masked token using the model
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            ###################change the best###################
            # Extract logits for the masked position
            logits = outputs.logits[0, mask_position]
            softmax = torch.nn.Softmax(dim=-1)
            probs = softmax(logits)

            # Find the most likely token ID
            predicted_token_id = probs.argmax().item()
            # Replace the token in the predicted sentence
            predicted_sentence_tokens[mask_position] = predicted_token_id

    # Decode the fully predicted sentence
    final_predicted_sentence = tokenizer.decode(predicted_sentence_tokens)
    print("Final predicted sentence:", final_predicted_sentence)

    return final_predicted_sentence


# Only pick the Middle probability in the Top-k

def Adversarial_sample_middle_topk(start_phrase, tokenizer, model):
    # Tokenization
    tokenized_phrase = tokenizer.encode(start_phrase, add_special_tokens=False)

    # Find the position of the first comma or period
    comma_index = tokenized_phrase.index(
        tokenizer.encode(',', add_special_tokens=False)[0]) if ',' in start_phrase else len(tokenized_phrase)
    period_index = tokenized_phrase.index(
        tokenizer.encode('.', add_special_tokens=False)[0]) if '.' in start_phrase else len(tokenized_phrase)

    # Determine the minimum index (first punctuation) and start after it
    start_index = min(comma_index, period_index) + 1  # +1 to start after the punctuation

    # POS tagging (assuming pos_detection function is defined elsewhere)
    pos_tags = pos_detection(tokenized_phrase, tokenizer)

    # To store the predicted words for each position
    predicted_sentence_tokens = tokenized_phrase[:]

    # Iterate over each token position, starting after the first punctuation
    for mask_position in range(start_index, len(tokenized_phrase)):
        # Check if the current word's POS is in the targeted POS list
        if pos_tags[mask_position][1] in {'VB', 'VBZ', 'VBD', 'VBN', 'JJ', 'NNS'}:
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

            # Calculate the middle index or indices
            middle_index = probs.shape[0] // 2
            if probs.shape[0] % 2 == 0:
                # Choose randomly between the two middle indices if even
                middle_indices = [middle_index - 1, middle_index]
                chosen_index = middle_indices[torch.randint(0, 2, (1,)).item()]
            else:
                # Directly choose the middle index if odd
                chosen_index = middle_index

            # Get the token ID of the middle ranked probability
            predicted_token_id = probs.sort(descending=True).indices[chosen_index].item()
            # Replace the token in the predicted sentence
            predicted_sentence_tokens[mask_position] = predicted_token_id

    # Decode the fully predicted sentence
    final_predicted_sentence = tokenizer.decode(predicted_sentence_tokens)
    print("Final predicted sentence:", final_predicted_sentence)

    return final_predicted_sentence

