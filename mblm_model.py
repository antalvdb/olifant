import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel
import transformers
import torch.nn.functional as F
import numpy as np

# Global verbosity level
VERBOSITY = 1

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

def pad_prompt_tokenids(token_ids, max_len=16, pad_token_id=None):
    """Pad or trim the list of token IDs to make it exactly `max_len`."""
    if token_ids is None:
        token_ids = []

    current_len = len(token_ids)

    if current_len < max_len:
        # Pad with pad_token_id if provided, otherwise use 0
        padding = [pad_token_id if pad_token_id is not None else 0] * (max_len - current_len)
        token_ids = padding + token_ids  # prepend padding for Timbl
        #token_ids = token_ids + padding  # append padding (more common)
    elif current_len > max_len:
        token_ids = token_ids[-max_len:]  # trim for Timbl

    return token_ids


def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 1, labels.unsqueeze(1)).squeeze(-1)
    return logp_label

class TimblHuggingFaceModel(PreTrainedModel):

    # Define a function to replace values with actual floats
    def float_converter(match):
        return f"{match.group(1)}: {float(match.group(2))}"

    def sequence_logprob(self, labels, tokenizer, max_len=16):
        with torch.no_grad():
            seq_log_prob = []
            pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) #Get the pad_token_id

            for i in range(len(labels[0])):
                # Pad the input token IDs directly
                padded_token_ids = pad_prompt_tokenids(labels[0][:i+1].tolist(), max_len=max_len, pad_token_id=pad_token_id)
                log(f"sequence_logprob: padded_token_ids: {padded_token_ids}", level = 3)

                # Convert padded token IDs to a PyTorch tensor
                input_ids = torch.tensor(padded_token_ids, dtype=torch.int64).unsqueeze(0).to("cpu")
                log(f"sequence_logprob: input_ids: {input_ids}", level = 3)

                output = self(input_ids)
                log(f"sequence_logprob: output: {output}", level = 3)
                next_token_logits = output.logits[0, :]
                log(f"sequence_logprob: next_token_logits: {next_token_logits}", level = 3)

                # Get the actual next token id from labels
                next_token_id = labels[0][i].unsqueeze(0)
                log(f"sequence_logprob: next_token_id: {next_token_id}", level = 3)

                # Calculate log probability of the actual next token
                # Added squeeze(0) to next_token_id to make it have shape [1]
                log_probs = log_probs_from_logits(next_token_logits.unsqueeze(0), next_token_id.squeeze(0).unsqueeze(0))
                log(f"sequence_logprob: log_probs: {log_probs}", level = 3)

                # Check if the log probability is -inf and skip if it is, log a warning
                if np.isinf(log_probs.cpu().numpy()):
                    log(f"Warning: log probability is -inf for token index: {i}, skipping.", level = 3)

                else:
                    seq_log_prob.append(log_probs.cpu().numpy().item()) # Append the log probability as a scalar
                    log(f"sequence_logprob: log_probs added: {log_probs}", level = 3)

                # Prepare for the next iteration, if it's not the last token
                if i < len(labels[0]) - 1:
                  input_ids = torch.cat((input_ids[0, 1:].unsqueeze(0), next_token_id.unsqueeze(0)), dim=1).unsqueeze(0)

        return np.sum(seq_log_prob) # Sum all values in the list to return a single float

    def __init__(self, config, timbl_classifier, tokenizer):
        super().__init__(config)
        self.timbl_classifier = timbl_classifier
        self.tokenizer = tokenizer  # Store tokenizer
        # Create an empty tensor for probabilities
        self.probabilities_tensor = torch.empty(0, device="cpu")
        
    def forward(self, input_ids, **kwargs):

        # Convert input_ids to Timbl format
        timbl_input = self.convert_to_timbl_input(input_ids)
        log(f"Timbl input: {timbl_input}",level=3)

        # Get Timbl predictions
        classlabel, distribution, distance = self.timbl_classifier.classify(timbl_input)
        log(f"Classlabel: {classlabel}", level = 3)
        log(f"Distribution: {distribution}", level = 3)
        log(f"Distance: {distance}", level = 3)
        # Convert Timbl output to Hugging Face format
        logits = self.convert_to_huggingface_logits(distribution)
        log(f"Logits: {logits}", level = 3)

        # Return logits and other relevant outputs
        return transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(logits=logits)

    def convert_to_timbl_input(self, input_ids):

        #print("inside convert_to_timbl_input")

        """Converts Hugging Face input_ids to Timbl input format."""
        # Decode input_ids to a string of tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        log(f"Tokens: {tokens}", level = 3)

        # Return the array of tokens directly
        return tokens

    def convert_to_huggingface_logits(self, distribution):

        # Bypassing the typical HuggingFace device setting and passing
        device = "cpu"

        # Get vocabulary size from the tokenizer
        vocab_size = self.tokenizer.vocab_size

        # Initialize logits with a default value (e.g., -inf)
        logits = torch.full((1, vocab_size), float('-inf'), device=device)

        # Filter out tokens not in the Hugging Face vocabulary
        hf_token_ids = [self.tokenizer.convert_tokens_to_ids(word)
                          for word in distribution.keys()]

        # Convert hf_token_ids to tensor
        hf_token_ids_tensor = torch.tensor(hf_token_ids, device=device)

        # Get probabilities from the distribution (This is the fix!)
        probabilities = [value for value in distribution.values()]

        # Resize the probabilities tensor after filtering
        self.probabilities_tensor.resize_(len(probabilities)).copy_(torch.tensor(probabilities))

        logits[0].scatter_(0, hf_token_ids_tensor, self.probabilities_tensor)  # In-place scatter

        return logits

    def custom_generate(self, input_ids, max_new_tokens, num_beams=1, do_sample=False, temperature=1.0, top_k=0, **kwargs):
        """
        Generates text using the Timbl model iteratively, with optional beam search and temperature.

        Args:
            input_ids: The input token IDs as a torch tensor.
            max_new_tokens: The maximum number of tokens to generate.
            num_beams: The number of beams for beam search (default is 1, which is greedy decoding).
            do_sample: If True, use temperature sampling, otherwise use greedy decoding or beam search.
            temperature: The temperature for sampling (default is 1.0).
            top_k:  The number of top tokens to consider during sampling.
            kwargs: Additional arguments (not currently used but kept for consistency).

        Returns:
            torch.Tensor: The generated sequence of token IDs.
        """
        batch_size = input_ids.shape[0]

        # Initialize variables for beam search
        if num_beams > 1 and not do_sample:

            #Create a list to store the sequences
            sequences = [input_ids.clone() for _ in range(num_beams)]

            # Create a list to store scores for the sequences
            sequence_scores = [torch.zeros(batch_size, device=input_ids.device) for _ in range(num_beams)]

        else:
            sequences = [input_ids.clone()]  # Start with the input ids

        # Tokenize the initial prompt and convert tokens back to words
        initial_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Generate padded instances
        padded_instances = []
        for i in range(len(initial_tokens)):
            # Take the tokens up to the current position and pad them
            instance = pad_prompt(initial_tokens[:i], max_len=16)
            padded_instances.append((instance, initial_tokens[i] if i < len(initial_tokens) else '_'))

        # Add instances to memory
        for input_instance, next_token in padded_instances:
            log(f"memorized from prompt: {input_instance} {next_token}", level=2)
            self.timbl_classifier.append(input_instance, next_token)

        
        with torch.no_grad():
             for _ in range(max_new_tokens):

                all_candidates = [] #Store all candidates in the beam search

                for i, seq in enumerate(sequences):
                    # Pad the input tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(seq[0])
                    padded_tokens = pad_prompt(tokens, max_len=16)
                    log(f"padded_tokens: {padded_tokens}", level = 3)

                    # Convert padded_tokens back into token_ids for timbl input
                    timbl_input_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
                    timbl_input_ids = torch.tensor(timbl_input_ids, dtype=torch.int64).unsqueeze(0).to("cpu")

                    # Get model output
                    outputs = self(timbl_input_ids)
                    logits = outputs.logits

                    #Apply log softmax on the logits
                    log_probs = F.log_softmax(logits[:, 1:], dim=-1)

                    if do_sample: #if temperature sampling is enabled
                      # Apply temperature scaling
                      scaled_logits = logits[:, 1:] / temperature

                      # Apply top-k filtering
                      if top_k > 0:
                            filter_value = -float('Inf')
                            top_k_values, _ = torch.topk(scaled_logits, top_k, dim=-1)
                            min_top_k = top_k_values[:, -1].unsqueeze(-1)  # Get the smallest top-k value
                            scaled_logits = torch.where(scaled_logits < min_top_k, torch.tensor(filter_value).to("cpu"), scaled_logits)

                      # Sample from the distribution
                      probabilities = torch.softmax(scaled_logits, dim=-1)
                      predicted_token_id = torch.multinomial(probabilities, num_samples=1) + 1  # sample
                      # Correct the unsqueeze dimension
                      sequences[i] = torch.cat((seq, predicted_token_id.unsqueeze(0).squeeze(0)), dim=1)

                    elif num_beams > 1: # If beam search is enabled
                        top_k_probs, top_k_ids = torch.topk(log_probs, num_beams, dim=-1)

                        #Prepare the candidate sequences
                        for j in range(num_beams):
                            candidate_seq = torch.cat((seq, top_k_ids[:,j].unsqueeze(0) + 1), dim=1)
                            candidate_score = sequence_scores[i] + top_k_probs[:,j] #accumulate the score
                            all_candidates.append((candidate_seq, candidate_score))
                    else: #if greedy decoding is enabled
                        predicted_token_id = torch.argmax(logits[:, 1:], dim=-1) + 1
                        sequences[i] = torch.cat((seq, predicted_token_id.unsqueeze(0)), dim=1)

                if num_beams > 1 and not do_sample:
                    #Select the top num_beams candidates based on score
                    ordered_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                    sequences = [seq for seq, _ in ordered_candidates[:num_beams]]
                    sequence_scores = [score for _, score in ordered_candidates[:num_beams]]

        return sequences[0]  # Return the generated sequence of token IDs
