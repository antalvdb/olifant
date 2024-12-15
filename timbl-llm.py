import timbl
from transformers import AutoTokenizer
import re
import time

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')

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
        print("Tokenization failed; 'initial_tokens' is None.")
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
        print("memorized: ", input_instance, next_token)
        classifier.append(input_instance, next_token)

    # Use the final part of the prompt for further generation
    prompt_words = pad_prompt(initial_tokens)
    
    generated_tokens = prompt_words[:]  # Store the full generated text

    try:
        # Loop until max words generated or a period token is found
        for _ in range(max_words):
            next_word = None

            classlabel, distribution, distance = classifier.classify( prompt_words )

            # add instance to instance base
            classifier.append(prompt_words, classlabel)
            
            print(prompt_words)
            print(classlabel)
            print(distribution)
            print(distance)
            
            generated_tokens.append(classlabel)
                
            # Shift prompt words and add the new word
            prompt_words = prompt_words[1:] + [classlabel]
                
            # Stop if a period is generated
            #if classlabel == ".":
            #    break
                
        # Detokenize the generated tokens
        generated_text = tokenizer.convert_tokens_to_string(generated_tokens)

        # Strip off original padding characters
        generated_text = generated_text.replace("_", "").strip()
        
        # Print the final generated text
        print("Generated text:", generated_text)
                    
    except Exception as e:
        print(f"Error: {e}")
        
# Example usage
classifier = timbl.TimblClassifier("gigacorpus-100k_tok.l16r0", "-a4 +D" )
classifier.load()

# Loop to continuously ask for input and classify
while True:
    # Take input from the user
    user_input = input("Please enter prompt (or type 'exit' to quit): ")
    
    # Check if the user wants to exit
    if user_input.lower() == 'exit':
        print("Exiting.")
        break
    
    # Pass the input to the classifier function
    generate_text_from_api(classifier, user_input)
