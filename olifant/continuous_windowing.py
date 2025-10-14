import sys

def generate_windowed_instances(file_path, window_size=16):
    # Start with an empty list to accumulate tokens for each block
    tokenized_text = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace from the line
            stripped_line = line.strip()
            
            # Check if the line is empty, indicating the end of a block
            if not stripped_line:
                # Process the accumulated tokens for the current block
                if tokenized_text:
                    # Pad the beginning of the tokenized text with underscores
                    padded_text = ["_"] * window_size + tokenized_text
                    
                    # Generate and print each windowed instance for this block
                    for i in range(window_size, len(padded_text) - 1):
                        context = padded_text[i - window_size:i]
                        target = padded_text[i]
                        print(f"{' '.join(context)} {target}")
                        
                        # Reset tokenized_text for the next block
                        tokenized_text = []
                        
            else:
                # Append tokens from the non-empty line to the current block
                tokenized_text.extend(stripped_line.split())
                    
        # Process any remaining tokens after the last line
        if tokenized_text:
            padded_text = ["_"] * window_size + tokenized_text
            for i in range(window_size, len(padded_text) - 1):
                context = padded_text[i - window_size:i]
                target = padded_text[i]
                print(f"{' '.join(context)} {target}")

def main():
    if len(sys.argv) != 2:
        print("Usage: olifant-continuous-windowing <filename>")
    else:
        file_path = sys.argv[1]
        generate_windowed_instances(file_path)

# Check if the filename argument is provided
if __name__ == "__main__":
    main()

