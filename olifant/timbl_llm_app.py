from flask import Flask, render_template_string, request
import socket
from transformers import AutoTokenizer
import re
import time
import threading

# Load the tokenizer (move outside the function for efficiency)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# --- Your existing functions (pad_prompt, generate_text_from_server, log) ---
# Include your pad_prompt, generate_text_from_server, and log functions here.
# I've omitted them for brevity in this example, but copy and paste them.
def log(message, level=1):
    """Logs a message if the verbosity level is sufficient."""
    # In a web app, you might want to log to a file or a proper logging system
    # instead of just printing to the console. For now, we'll keep print.
    print(message)

def pad_prompt(words, max_len=16):
    """Pad or trim the list of words to make it exactly `max_len` words."""
    if words is None:
        words = []
    if len(words) < max_len:
        words = ['_'] * (max_len - len(words)) + words
    else:
        words = words[-max_len:]
    return words

def generate_text_from_server(host, port, initial_prompt, max_words=200):
    # Tokenize the initial prompt and convert tokens back to words
    initial_tokens = tokenizer.tokenize(initial_prompt)

    if initial_tokens is None:
        log("Tokenization failed; 'initial_tokens' is None.", level=1)
        initial_tokens = []

    # Prepare the initial prompt, padded or trimmed to 8 words
    prompt_words = pad_prompt(initial_tokens)

    generated_tokens = prompt_words[:]  # Store the full generated text
    generated_text = "" # Initialize generated_text

    try:
        # Create a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # Connect to the server
            client_socket.connect((host, port))

            # Receive the initial "Welcome" message from the server
            welcome_message = client_socket.recv(1024).decode('utf-8')
            log(f"Received welcome message: {welcome_message}", level=3)

            # Loop until max words generated or a period token is found
            for _ in range(max_words):
                next_word = None

                log(f"Current prompt: {' '.join(prompt_words)}", level=3)

                # Keep asking the server until a valid next word is received
                while not next_word:
                    # Prepare the message to send with "??" appended
                    message = "classify " + " ".join(prompt_words) + " ??\n"
                    client_socket.sendall(message.encode('utf-8'))

                    time.sleep(0.01) # Reduced sleep

                    # Receive the server response
                    data = client_socket.recv(1024).decode('utf-8')

                    log(f"Received data: {data}", level=3)

                    # Check for the CATEGORY line in the response
                    lines = data.strip().splitlines()
                    for line in lines:
                        if line.startswith("CATEGORY "):
                            # Extract the word inside curly brackets
                            match = re.search(r"\{(.*?)\}", line)
                            if match:
                                next_word = match.group(1)  # Extract the predicted word inside `{}`

                # Add the predicted word to the generated text
                generated_tokens.append(next_word)

                log(f"Predicted word: {next_word}", level=3)

                # Shift prompt words and add the new word
                prompt_words = prompt_words[1:] + [next_word]

                # Stop if a period is generated (optional, uncomment if needed)
                # if next_word == ".":
                #     break

        # Detokenize the generated tokens
        generated_text = tokenizer.convert_tokens_to_string(generated_tokens)

        # Strip off original padding characters
        generated_text = generated_text.replace("_", "").strip()

        log(f"Generated text: {generated_text}", level=1)

    except Exception as e:
        generated_text = f"Error: {e}"
        log(f"Error: {e}", level=1)


    return generated_text


# Flask Web Application
app = Flask(__name__)

# Simple HTML template for the web page
# Simple HTML template with basic CSS styling
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Text Generation</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            margin: 20px auto;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box; /* Include padding and border in the element's total width */
        }
        input[type="submit"] {
            background-color: #5cb85c;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #4cae4c;
        }
        h2 {
            color: #333;
            margin-top: 30px;
        }
        p {
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            margin: 10px auto;
            white-space: pre-wrap; /* Preserve whitespace and wrap lines */
        }
    </style>
</head>
<body>
    <h1>Text Generation</h1>
    <form method="post">
        <label for="prompt">Enter your initial prompt:</label><br>
        <input type="text" id="prompt" name="prompt" size="50"><br><br>
        <input type="submit" value="Generate Text">
    </form>

    {% if generated_text %}
    <h2>Generated Text:</h2>
    <p>{{ generated_text }}</p>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = None
    if request.method == 'POST':
        initial_prompt = request.form['prompt']
        # Example usage (you'll need to set your server host and port)
        host = 'localhost'  # Server's IP address or hostname
        port = 8888              # Port number on the server

        # Generate text in a separate thread to avoid blocking the web server
        # In a production environment, you would use a task queue like Celery
        # For a simple example, threading is acceptable.
        def generate_and_store(prompt):
            nonlocal generated_text # Access the generated_text variable from the outer scope
            generated_text = generate_text_from_server(host, port, prompt)

        thread = threading.Thread(target=generate_and_store, args=(initial_prompt,))
        thread.start()
        thread.join() # Wait for the generation to complete (consider making this asynchronous for a better user experience)


    return render_template_string(HTML_TEMPLATE, generated_text=generated_text)

def main():
    app.run(host='0.0.0.0', port=8001, debug=True) # debug=True is good for development

# This part is crucial for running the app when the file is executed
if __name__ == '__main__':
    main()
