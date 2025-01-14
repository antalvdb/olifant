from transformers import AutoTokenizer
import re
import time
import argparse
import sys

sys.path.append('timbl-python')
import timbl

# Global verbosity level
VERBOSITY = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')

def log(message, level=1):
    """Logs a message if the verbosity level is sufficient."""
    if VERBOSITY >= level:
        print(message)

def pad_prompt(words, max_len=16):
    """Pad or trim the list of words to make it exactly `max_len` words."""
    if words is None:
        words = []  # Ensure words is a list
    if len(words) < max_len:
        words = ['_'] * (max_len - len(words)) + words
    else:
        words = words[-max_len:]
    return words

def generate_text_from_api(classifier, initial_prompt, max_words=200):
    # Tokenize the initial prompt and convert tokens back to words
    initial_tokens = tokenizer.tokenize(initial_prompt)
    
    if initial_tokens is None:
        log("Tokenization failed; 'initial_tokens' is None.", level=1)
        initial_tokens = []                

    # Prepare the initial prompt, padded or trimmed to 16 words
    padded_instances = []

    # Generate padded instances for next-token predictions
    for i in range(len(initial_tokens)):
        # Take the tokens up to the current position and pad them
        instance = pad_prompt(initial_tokens[:i], max_len=16)
        padded_instances.append((instance, initial_tokens[i] if i < len(initial_tokens) else '_'))
        
    # Add instances to memory
    for input_instance, next_token in padded_instances:
        log(f"memorized: {input_instance} {next_token}", level=2)
        classifier.append(input_instance, next_token)

    # Use the final part of the prompt for further generation
    prompt_words = pad_prompt(initial_tokens)
    
    generated_tokens = prompt_words[:]  # Store the full generated text

    try:
        # Loop until max words generated or a period token is found
        for _ in range(max_words):
            next_word = None

            classlabel, distribution, distance = classifier.classify(prompt_words)

            # Add instance to instance base
            classifier.append(prompt_words, classlabel)
            
            log(f"Prompt words: {prompt_words}", level=2)
            log(f"Classlabel: {classlabel}", level=2)
            log(f"Distribution: {distribution}", level=3)
            log(f"Distance: {distance}", level=3)
            
            generated_tokens.append(classlabel)
                
            # Shift prompt words and add the new word
            prompt_words = prompt_words[1:] + [classlabel]
                
            # Stop if a period is generated
            # if classlabel == ".":
            #     break
                
        # Detokenize the generated tokens
        generated_text = tokenizer.convert_tokens_to_string(generated_tokens)

        # Strip off original padding characters
        generated_text = generated_text.replace("_", "").strip()
        
        # Print the final generated text
        log(f"Generated text: {generated_text}", level=1)
                    
    except Exception as e:
        log(f"Error: {e}", level=1)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Memory-based text generator")
    parser.add_argument("--classifier", type=str, required=True, help="Path to the Timbl classifier file")
    parser.add_argument("--timbl_args", type=str, required=True, help="Timbl arguments as a single string (e.g., '-a4 +D')")
    parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level (0: silent, 1: basic, 2: detailed, 3: debug)")
    args = parser.parse_args()

    # Set global verbosity level
    VERBOSITY = args.verbosity

    # Initialize the classifier
    classifier = timbl.TimblClassifier(args.classifier, args.timbl_args)
    classifier.load()

    # Loop to continuously ask for input and classify
    while True:
        # Take input from the user
        user_input = input("Please enter prompt (or type 'exit' to quit): ")
        
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            log("Exiting.", level=1)
            break
        
        # Pass the input to the classifier function
        generate_text_from_api(classifier, user_input)
