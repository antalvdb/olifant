# Memory-based language modeling

This repository contains instructions and code to install, train and run memory-based LLMs. 

Looking for an LLM that is relatively eco-friendly? MBLMs rely on CPUs. 
No GPUs or TPUs are required for training or inference.
Training MBLMs is costly in terms of RAM, but not in terms of time or computing resources.
Running an MBLM in autoregressive GPT-style mode also costs RAM, but still relies on CPUs and is reasonably fast as well, depending on the selected
approximation of k-nearest neighbor classification.


## Installation

MBLM relies on [python3-timbl](https://github.com/proycon/python-timbl) and the [TiMBL](https://github.com/LanguageMachines/timbl/) memory-based classification engine.

For training, the command-line version of TiMBL is required. Install TiMBL on Debian/Ubuntu systems with

``% apt install timbl``

On macOS with brew, invoke

``% brew install timbl``

Next, for inference, install the Python bindings to TiMBL, [python3-timbl](https://github.com/proycon/python-timbl), with pip (wheels are available for Python versions 3.10, 3.11, and 3.12 on systems with glibc 2.28 or higher; on macOS, installation only works with Python 3.13 currently):

``% pip install python3-timbl``

## Usage

### Tokenization

Training MBLM assumes that you have a tokenizer and a raw-text training set `textfile`. The tokenizer will have to be the same tokenizer used for testing.
First, the text is tokenized:

``% python3 tok.py textfile``

This creates a file `textfile_tok` which then needs to be converted to a fixed-width instance base to make it suitable training data for TiMBL.
The example works with an input buffer of 16 tokens, which in current LLM terms is a very small input buffer. 
At inference time, however, single instances are incrementally stored in memory, becoming available for the next steps in inference in the internal "long-term" memory of the memory-based classifier.

``% python3 continuous-windowing.py textfile_tok > textfile_tok.l16r0``

This creates `textfile_tok.l16r0`, creating 16-token windowed instances with the next token as the label to be classified and all previous tokens as context.
Empty lines in the original tokenized text signify the reset of the context window (padded with "_").

### Training

Training can then be invoked by calling TiMBL. This can take a while and may consume high amounts of RAM.

``% timbl -f textfile_tok.l16r0 -a0 +D -I textfile_tok.l16r0.ibase``

The end result is `textfile_tok.l16r0.ibase`, an indexed and compressed instance base suitable for TiMBL classification. In LLM terms, this is the model file
that you will need for your favorite LLM inference steps.

The option `-a0` means that the training set is compressed losslessly, with compression rates around 10-30%. 
With `-a1`, a strong lossy compression is applied, yielding higher compression levels around 90-95%, and considerably faster but less accurate inference.

### Fine-tuning

MBLMs are natural incremental learners, so any learned model can be complemented by additional fine-tuning from any new training set, creating a new `ibase` model. 
This requires a TiMBL invocation similar to the training command; it now includes a previously generated `ibase` model file as starting point. Assuming you
have tokenized and windowed a new training set `finetune_tok`:

``% timbl -a0 +D --clones=16 -i textfile_tok.l16r0.ibase -f finetune_tok.l16r0 -I textfile-finetune_tok.l16r0.ibase``

Choose your own naming conventions to keep track of trained and finetuned `ibase` model files. Any `ibase` file can be the starting point for further finetuning.
This also offers a way to do stepwise training with segments of training data under limited RAM conditions.

### Inference

Simple GPT-style text completion can be invoked by issuing

``% python3 timbl-llm.py --classifier textfile-finetune_tok.l16r0 --tokenizer bert-base-cased --timbl_args '-a4 +D' --verbosity 3``

This call assumes the presence of `textfile-finetune_tok.l16r0.ibase`. The arguments passed to the TiMBL engine are '-a4 +D', 
invoking the so-called TRIBL2 k-NN approximation. See the [TiMBL reference guide](https://github.com/LanguageMachines/timbl/blob/master/docs/Timbl_6.4_Manual.pdf) 
for all possible algorithmic variants (-a), the important k parameter (set to 1 by default), and many more options.

You can also run a Jupyter Notebook version:

``% jupyter notebook timbl-llm.ipynb``

Be sure to adjust the way you load your `.ibase` model file.

### Inference, Hugging Face style

In this Jupyter Notebook you see how MBLM can be run Hugging
Face style:

``% jupyter notebook timbl-llm-hf.ipynb``

An excerpt from this code shows how a `TimblHuggingFaceModel` is initialized:

```
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Initialize the Timbl classifier
    classifier = timbl.TimblClassifier(args.classifier, args.timbl_args)
    classifier.load()

    config = AutoConfig.from_pretrained("antalvdb/mblm-chatbot-instruction-prompts-igtree")
    tokenizer.add_special_tokens({'pad_token': '_'})
    tokenizer.pad_token = "_"

    # Initialize the TimblHuggingFaceModel
    model = TimblHuggingFaceModel(config, classifier, tokenizer)
```

### Credits

TiMBL was created 25 years ago by a team that was once the Induction of Linguistic Knowledge group at 
Tilburg University, the Netherlands; members moved to the Computational Linguistics, Psycholinguistics and Sociolinguistics
group at Antwerp University, Belgium, and the Centre for Language and Speech Technology at Radboud University, Nijmegen, 
the Netherlands. Core developer of TiMBL is Ko van der Sloot. Other contributors were Walter Daelemans, Antal van den Bosch, Jakub Zavrel, Peter Berck,
Maarten van Gompel, and many more people credited fully in the [TiMBL reference guide](https://github.com/LanguageMachines/timbl/blob/master/docs/Timbl_6.4_Manual.pdf).

MBLM is a re-implementation of WOPR, a C++ version of a TiMBL-based word predictor developed by Peter Berck,
funded under the NWO Vici project "Memory Models of Language" (2006-2011) awarded to
Antal van den Bosch. Peter Berck wrote a [PhD thesis](https://repository.ubn.ru.nl/bitstream/handle/2066/168708/168708.pdf?sequence=1) on the topic. 
Later, work on memory-based word prediction was
carried forwards by Wessel Stoop ([Valkuil](https://valkuil.net), ) and Maarten van Gompel ([Colibri Core](https://github.com/proycon/colibri-core)).
See this [interactive publication](https://pudding.cool/2019/04/text-prediction/) on autocompletion and next-word prediction.
