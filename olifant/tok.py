import sys
import os
import pandas as pd
from transformers import AutoTokenizer

def process_file(input_filename):
    # Generate the output filename by adding "tok" before the file extension
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_tok{ext}"

    # Read the input file
    with open(input_filename, 'r') as file:
        lines = file.readlines()
    
    # Create a DataFrame
    df = pd.DataFrame(lines, columns=['text'])
    #print("DataFrame head after reading file:")
    #print(df.head())

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Tokenize the text
    df['tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
    #print("DataFrame head after tokenization:")
    #print(df.head())

    # Write the tokens to the output file
    with open(output_filename, 'w') as file:
        for tokens in df['tokens']:
            # Join the tokens list into a single string and write to the file
            file.write(' '.join(tokens) + '\n')

    #print(f"Processed file saved as {output_filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: olifant-tok <input_filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    process_file(input_filename)

if __name__ == "__main__":
    main()

#with open('questions.txt', 'r') as file:
#    lines = file.readlines()

#df = pd.DataFrame(lines, columns=['text'])
#print(df.head())

#tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
#df['tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
#print(df.head())

#with open('questions_tok.txt', 'w') as file:
#    for tokens in df['tokens']:
# Join the tokens list into a single string and write to the file
#        file.write(' '.join(tokens) + '\n')
