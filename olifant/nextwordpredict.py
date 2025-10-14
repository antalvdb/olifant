device = "cpu" # Set a default device for notebook execution

import os
import time
import torch
import timbl

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from olifant.mblm_model import TimblHuggingFaceModel, log,  pad_prompt_tokenids, log
from codecarbon import track_emissions

def evaluate_mblm_next_word_prediction(model, tokenizer, text_filepath, device):
    """Evaluates MBLM on next-word prediction using raw text from a JSON file.

    Args:
        model: The MBLM model.
        tokenizer: The tokenizer used for the model.
        text_filepath: Path to the JSON file containing raw text.
        device: The device to run the evaluation on (e.g., "cpu", "cuda").

    Returns:
        The accuracy of the model on next-word prediction.
    """
    # --- Tokenization step (performed only once) ---

    # Get the total number of lines in the raw text file
    with open(text_filepath, "r") as f:
        total_lines = sum(1 for _ in f)
    print(f"# lines to be processed: {total_lines}")

    # Define tokenized_text_filepath based on text_filepath
    tokenized_text_filepath = text_filepath + ".tokenized"
    # Check if the tokenized file already exists
    if not os.path.exists(tokenized_text_filepath):
        print("Tokenizing raw text file...")
        with open(text_filepath, "r") as f_in, open(tokenized_text_filepath, "w") as f_out: # Change raw_text_filepath to text_filepath
            for line in tqdm(f_in, total=total_lines, desc="Tokenizing lines"):
                tokenized_line = tokenizer.tokenize(line)
                f_out.write(" ".join(tokenized_line) + "\n")  # Write tokenized line to file
        print("Tokenization complete. Tokenized file saved to:", tokenized_text_filepath)
    else:
        print("Using existing tokenized file:", tokenized_text_filepath)

    num_correct = 0
    num_total = 0
    window_size = 16  # Set the sliding window size

    # Start timer
    start_time = time.time()

    # Open the text file and read the content line by line
    with open(tokenized_text_filepath, "r") as f:
        for _ in tqdm(range(total_lines), desc="Processing lines"):  # Progress bar for lines
            line = f.readline()  # Read one line at a time

            # Split the line into tokens
            tokenized_text = line.split()

            # Process the tokenized text for next-word prediction within the line
            for i in range(len(tokenized_text) - 1):
                # Get the current context and the next word (target)
                # Use sliding window for context
                context = tokenized_text[max(0, i - window_size + 1) : i + 1]
                target_word = tokenized_text[i + 1]

                log(f"Context: {' '.join(context)}", level=3)
                log(f"Target word: {target_word}", level=3)

                # Convert context to input IDs
                input_ids = tokenizer.convert_tokens_to_ids(context)

                # Pad the input_ids using pad_prompt_tokenids
                padded_input_ids = pad_prompt_tokenids(
                    input_ids, max_len=16, pad_token_id=tokenizer.pad_token_id
                )  # Pad to 16 tokens, using tokenizer's pad_token_id

                # Convert padded_input_ids to a PyTorch tensor
                input_ids = torch.tensor([padded_input_ids], dtype=torch.long).to(device)

                # Get MBLM's prediction for the next word
                logits = model(input_ids).logits

                # Reverted to using torch.argmax:
                predicted_word_id = torch.argmax(logits[0, :]).item()  # Get the predicted word ID

                # Convert the predicted word ID to a token
                predicted_word = tokenizer.convert_ids_to_tokens(predicted_word_id)

                log(f"Predicted word: {predicted_word}", level=3)

                # Compare the prediction with the actual word
                if predicted_word == target_word:
                    num_correct += 1

                num_total += 1

    # End timer
    end_time = time.time()

    # Calculate accuracy, handling division by zero
    accuracy = num_correct / num_total if num_total else 0.0

    # Calculate predictions per second
    predictions_per_second = num_total / (end_time - start_time)

    return accuracy, num_correct, num_total, predictions_per_second

def main():
    # Initialize the tokenizer
    print("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    # Initialize the Timbl classifier
    print("Initializing classifier")
    classifier = timbl.TimblClassifier('edufineweb_train_000001-10k_tok.l16r0', '-a4 +D')
    classifier.load()

    print("Loading config")
    config = AutoConfig.from_pretrained("antalvdb/mblm-chatbot-instruction-prompts-igtree")
    tokenizer.add_special_tokens({'pad_token': '_'})
    tokenizer.pad_token = "_"

    @track_emissions(project_name="edufineweb_val")
    def test_model():
        accuracy, num_correct, num_total, predictions_per_second = evaluate_mblm_next_word_prediction(model, tokenizer, text_filepath, device)
        return accuracy, num_correct, num_total, predictions_per_second

    accuracy, num_correct, num_total, predictions_per_second = test_model()
    print(f"\n\nNext-token prediction accuracy: {accuracy} ({num_correct} out of {num_total})")
    print(f"Token predictions per second: {predictions_per_second:.2f}")

    # Initialize the TimblHuggingFaceModel
    model = TimblHuggingFaceModel(config, classifier, tokenizer)

    # Specify your raw untokenized text file used for testing
    text_filepath = "edufineweb_val_000000-10k.txt"
    
if __name__ == "__main__":
    main()
