# Memory-based language modeling

This repository contains instructions and code to install, train and run memory-based LLMs. 

Looking for an LLM that is relatively eco-friendly? MBLMs rely on CPU computing and (possibly lots of) RAM. No GPUs or TPUs required.
Training MBLMs is costly in terms of RAM, but not in terms of time or computing resources.
Running an MBLM in GPT-style mode also costs RAM, but still relies on CPUs and is reasonably fast as well, depending on the selected
approximation of k-nearest neighbor classification.


## Installation

MBLM relies on the TiMBL memory-based classification engine and python3-timbl. To install the engine, [TiMBL](https://github.com/LanguageMachines/timbl/),
invoke your platform's package manager. On Ubuntu/Debian Linux, do

``apt-get install timbl``

On macOS with brew, invoke

``brew install timbl``

Next, install its associated python bindings, [python3-timbl](https://github.com/proycon/python-timbl) (currently this only works for Ubuntu/Debian Linux):

``% pip install python3-timbl``

## Usage

### Tokenization

Training MBLM assumes that you have a tokenizer and a raw-text training set `textfile`. The tokenizer will have to be the same tokenizer used for testing.
First, the text is tokenized:

``python3 tok.py textfile``

This creates a file `textfile_tok` which then needs to be converted to a fixed-width instance base to make it suitable training data for TiMBL:

``python3 continuous-windowing.py textfile_tok > textfile_tok.l16r0``

This creates `textfile_tok.l16r0`, creating 16-token windowed instances with the next token as the label to be classified and all previous tokens as context. 
Empty lines in the original tokenized text signify the reset of the context window (padding with "_").

### Training

Training can then be invoked by calling TiMBL. This can take a while and may consume high amounts of RAM.

``timbl -f textfile_tok.l16r0 -a0 +D -I textfile_tok.l16r0.ibase``

The end result is `textfile_tok.l16r0.ibase`, an indexed and compressed instance base suitable for TiMBL classification. In LLM terms, this is the model file
that you will need for your favorite LLM inference steps.

### Fine-tuning

MBLMs are natural incremental learners, so any learned model can be complemented by additional fine-tuning from any new training set, creating a new `ibase` model. 
This requires a TiMBL invocation similar to the training command; it now includes a previously generated `ibase` model file as starting point. Assuming you
have tokenized and windowed a new training set `finetune_tok`:

``timbl -a0 +D --clones=16 -i textfile_tok.l16r0.ibase -f finetune_tok.l16r0 -I textfile-finetune_tok.l16r0.ibase``

Choose your own naming conventions to keep track of trained and finetuned `ibase` model files. Any `ibase` file can be the starting point for further finetuning.
This also offers a way to do stepwise training with segments of training data under limited RAM conditions.

### Inference

Simple GPT-style text completion can be invoked by issuing

``python3 timbl-llm.py``

The code will be updated to accept command-line parameters. Right now, the ibase file needs to be edited inside `timbl-llm.py`. 

### Credits

TiMBL was created 25 years ago by a team that was once the Induction of Linguistic Knowledge group at 
Tilburg University, the Netherlands; members moved to the Computational Linguistics, Psycholinguistics and Sociolinguistics
group at Antwerp University, Belgium, and the Centre for Language and Speech Technology at Radboud University, Nijmegen, 
the Netherlands. Core developer of TiMBL is Ko van der Sloot. Other contributors were Walter Daelemans, Antal van den Bosch, Jakub Zavrel, Peter Berck,
Maarten van Gompel, and many more people credited more fully in the TiMBL reference guide.

MBLM is a re-implementation of WOPR, a C++ version of a TiMBL-based word predictor developed by Peter Berck,
funded under the NWO Vici project "Memory Models of Language" (2006-2011) awarded to
Antal van den Bosch.